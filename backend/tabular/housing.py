"""Time-ordered apartment valuation and a reproducible demo market."""
from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.core.forecast import Apartment, Sale, SeriesInput, TabularInput
from backend.timeseries.classical import holt
from backend.timeseries.preprocessing import Series, prepare

DISTRICTS = (
    "Вахитовский", "Советский", "Приволжский", "Ново-Савиновский",
    "Московский", "Кировский", "Авиастроительный",
)
FIELDS = ("area_total", "floor", "floors_total", "build_year", "district", "house_type", "condition")


@dataclass(frozen=True)
class Market:
    apartment: Apartment
    sales: list[Sale]
    index: Series
    demo: bool
    external_index: bool


def demo_sales(seed: int = 43, months: int = 60, per_month: int = 25) -> list[Sale]:
    rng = np.random.default_rng(seed)
    start = 2020 * 12
    sales = []
    for month in range(months):
        year, month_of_year = divmod(start + month, 12)
        market_price = 82500 * 1.009 ** month * (1 + .035 * np.sin(2 * np.pi * month / 12))
        for _ in range(per_month):
            area = float(np.clip(rng.normal(55, 20), 16, 140))
            district = int(rng.integers(len(DISTRICTS)))
            floors = int(rng.choice([5, 9, 10, 16, 25]))
            floor = int(rng.integers(1, floors + 1))
            built = int(rng.integers(1955, year))
            condition = int(rng.integers(3))
            price = area * market_price * (1.18 - .043 * district + .0019 * (built - 1985) +
                                           .06 * condition - .035 * (floor == 1))
            price *= float(rng.lognormal(0, .08))
            sales.append(Sale(
                date=date(year, month_of_year + 1, int(rng.integers(1, 28))),
                district=DISTRICTS[district], area_total=round(area, 2), floor=floor,
                floors_total=floors, build_year=built, house_type="panel" if floors <= 10 else "monolith",
                condition=["needs_repair", "cosmetic", "euro"][condition], price_rub=round(price, 2),
            ))
    return sorted(sales, key=lambda row: row.date)


def from_sales(sales: list[Sale]) -> Series:
    groups: dict[date, list[float]] = {}
    for row in sales:
        groups.setdefault(row.date.replace(day=1), []).append(row.price_rub / row.area_total)
    return prepare(SeriesInput(
        observations=[{"date": month, "value": float(np.median(prices))} for month, prices in groups.items()],
        frequency="MS", nonnegative=True,
    ))


def prepare_market(data: TabularInput) -> Market:
    sales = sorted(data.market, key=lambda row: row.date) if data.market is not None else demo_sales()
    index = prepare(data.market_index) if data.market_index is not None else from_sales(sales)
    if index.frequency != "MS" or np.any(index.values <= 0):
        raise ValueError("Индекс рынка должен быть положительным месячным рядом")
    if (index.index[0].date() > sales[0].date.replace(day=1)
        or index.index[-1].date() < sales[-1].date.replace(day=1)):
        raise ValueError("Индекс рынка должен покрывать все месяцы истории квартир")
    return Market(data.apartment, sales, index, data.market is None, data.market_index is not None)


@dataclass(frozen=True)
class Encoder:
    centers: np.ndarray
    scales: np.ndarray
    categories: list[list[str]]
    groups: list[slice]

    @staticmethod
    def numeric(row: Apartment) -> list[float]:
        return [row.area_total, float(row.floor), float(row.floors_total),
                float(row.build_year) if row.build_year is not None else np.nan]

    @staticmethod
    def categorical(row: Apartment) -> list[str]:
        return [row.district.casefold(), (row.house_type or "unknown").casefold(),
                (row.condition or "unknown").casefold()]

    @classmethod
    def fit(cls, rows: list[Apartment]) -> "Encoder":
        x = np.asarray([cls.numeric(row) for row in rows])
        centers = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                            for col in x.T])
        x = np.where(np.isnan(x), centers, x)
        categories = [sorted(value for value, _ in Counter(column).most_common(64))
                      for column in zip(*(cls.categorical(row) for row in rows))]
        groups = [slice(i, i + 1) for i in range(4)]
        offset = 4
        for column in categories:
            groups.append(slice(offset, offset + len(column)))
            offset += len(column)
        return cls(centers, np.maximum(x.std(axis=0), 1e-6), categories, groups)

    def transform(self, rows: list[Apartment]) -> np.ndarray:
        x = np.asarray([self.numeric(row) for row in rows])
        x = (np.where(np.isnan(x), self.centers, x) - self.centers) / self.scales
        encoded = [[float(value == category) for value, options in
                    zip(self.categorical(row), self.categories) for category in options] for row in rows]
        return np.c_[x, np.asarray(encoded)]


def _learn(x: np.ndarray, y: np.ndarray, name: str, arguments: dict):
    if name == "fit_ensemble":
        weight = arguments["ridge_weight"]
        ridge = _learn(x, y, "fit_ridge", {"alpha": 8.})
        neighbors = _learn(x, y, "fit_neighbors", {"neighbors": 25})
        boost = _learn(x, y, "fit_boost", {"rounds": 32})
        return lambda matrix: (weight * ridge(matrix) + (1 - weight) / 2 *
                               (neighbors(matrix) + boost(matrix)))
    baseline = float(y.mean())
    if name == "fit_ridge":
        weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * arguments["alpha"],
                                  x.T @ (y - baseline))
        return lambda matrix: baseline + matrix @ weights
    if name == "fit_neighbors":
        neighbors = min(len(y), arguments["neighbors"])

        reference_norms = (x ** 2).sum(axis=1)

        def predict(matrix):
            predictions = []
            for start in range(0, len(matrix), 64):
                batch = matrix[start:start + 64]
                distances = np.maximum(0, (batch ** 2).sum(axis=1)[:, None] +
                                       reference_norms - 2 * batch @ x.T)
                closest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
                weights = 1 / np.maximum(np.take_along_axis(distances, closest, axis=1), .05)
                predictions.append((weights * y[closest]).sum(axis=1) / weights.sum(axis=1))
            return np.concatenate(predictions)

        return predict

    # Gradient boosting with depth-one regression trees.
    trees = []
    fitted = np.full(len(y), baseline)
    splits = []
    for column, values in enumerate(x.T):
        for threshold in np.unique(np.quantile(values, [.15, .35, .55, .75, .90])):
            mask = values <= threshold
            if 0 < mask.sum() < len(y):
                splits.append((column, threshold, mask))
    if splits:
        matrix = np.column_stack([mask for _, _, mask in splits]).astype(float)
        counts = matrix.sum(axis=0)
        for _ in range(arguments["rounds"]):
            residual = y - fitted
            left = matrix.T @ residual
            right = residual.sum() - left
            best = int(np.argmax(left ** 2 / counts + right ** 2 / (len(y) - counts)))
            column, threshold, mask = splits[best]
            low = .15 * left[best] / counts[best]
            high = .15 * right[best] / (len(y) - counts[best])
            fitted += np.where(mask, low, high)
            trees.append((column, threshold, low, high))

    def predict(matrix):
        result = np.full(len(matrix), baseline)
        for column, threshold, low, high in trees:
            result += np.where(matrix[:, column] <= threshold, low, high)
        return result

    return predict


def _market_prices(index: Series, rows: list[Sale]) -> np.ndarray:
    locations = index.index.searchsorted(pd.to_datetime([row.date.replace(day=1) for row in rows]),
                                         side="right") - 1
    return index.values[np.maximum(0, locations)]


def fit_market(market: Market, rows: list[Sale], name: str, arguments: dict):
    encoder = Encoder.fit(rows)
    index = market.index if market.external_index else from_sales(rows)
    targets = np.log(np.array([row.price_rub for row in rows]) /
                     _market_prices(index, rows))
    predict = _learn(encoder.transform(rows), targets, name, arguments)
    return encoder, predict, index


def backtest(market: Market, name: str, arguments: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cutoff = market.sales[int(len(market.sales) * .8)].date.replace(day=1)
    training = [row for row in market.sales if row.date < cutoff]
    test = [row for row in market.sales if row.date >= cutoff]
    if len(training) < 30 or len(test) < 5:
        raise ValueError("Нужна история продаж с отдельными месяцами для проверки")
    validation_market = market
    if market.external_index:
        n = market.index.index.searchsorted(pd.Timestamp(cutoff))
        if n < 12:
            raise ValueError("До начала проверки нужно 12 месяцев индекса")
        index = market.index
        validation_market = Market(market.apartment, market.sales,
                                   Series(index.index[:n], index.values[:n], index.frequency,
                                          index.period, True, index.missing_rate),
                                   market.demo, True)
    encoder, predict, index = fit_market(validation_market, training, name, arguments)
    dates = [row.date.replace(day=1) for row in test]
    last_month = index.index[-1].date()
    steps = [(day.year - last_month.year) * 12 + day.month - last_month.month for day in dates]
    projected = np.maximum(index.values[-1] * .2, holt(index.values, max(steps), .4, .12, .92))
    predictions = np.exp(predict(encoder.transform(test))) * projected[np.array(steps) - 1]
    return (np.array([row.price_rub for row in test]), predictions,
            np.array([row.price_rub for row in training]))


def shap_values(encoder: Encoder, predict, apartment: Apartment, reference: Apartment,
                index_value: float) -> list[dict]:
    """Exact interventional Shapley values over the seven input field groups."""
    from math import comb

    observed, baseline = encoder.transform([apartment, reference])
    n = len(FIELDS)
    matrix = np.repeat(baseline[None, :], 2 ** n, axis=0)
    masks = np.arange(2 ** n)
    for i, group in enumerate(encoder.groups):
        matrix[(masks & (1 << i)) != 0, group] = observed[group]
    prices = index_value * np.exp(predict(matrix))
    contributions = []
    for i, field in enumerate(FIELDS):
        subsets = [mask for mask in range(2 ** n) if not mask & (1 << i)]
        contribution = sum((prices[mask | (1 << i)] - prices[mask]) /
                           (n * comb(n - 1, mask.bit_count())) for mask in subsets)
        contributions.append({"feature": field, "impact_rub": round(float(contribution), 2)})
    return sorted(contributions, key=lambda row: abs(row["impact_rub"]), reverse=True)[:5]


def forecast(market: Market, horizon: int, name: str, arguments: dict) -> dict:
    encoder, predict, index = fit_market(market, market.sales, name, arguments)
    projected = np.maximum(index.values[-1] * .2, holt(index.values, horizon, .4, .12, .92))
    ratio = float(np.exp(predict(encoder.transform([market.apartment]))[0]))
    prices = projected * ratio
    reference = min(market.sales, key=lambda row: abs(row.area_total - 55) + abs(row.floor - 4))
    return {"predicted_price": float(prices[-1]), "price_per_m2": float(prices[-1] / market.apartment.area_total),
            "market_index_forecast": projected.tolist(), "predictions": prices.tolist(),
            "dates": pd.date_range(index.index[-1], periods=horizon + 1, freq="MS")[1:].strftime("%Y-%m-%d").tolist(),
            "factors": shap_values(encoder, predict, market.apartment, reference, float(projected[-1])),
            "data_origin": "synthetic_kazan_demo" if market.demo else "uploaded_sales"}
