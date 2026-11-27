"""Модель цены: LightGBM по характеристикам объекта + индекс рынка (тренд по времени).

Схема:
  1. Индекс рынка по месяцам убирает из цены общий рост/падение рынка.
  2. LightGBM учит «относительную» цену объекта: ln(цена/м²) − индекс месяца.
  3. Прогноз = LightGBM + прогноз индекса на нужный месяц (линейный тренд).
Оценка честная по времени: обучаем на прошлом, проверяем на самых новых месяцах.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.core.exceptions import DataValidationError
from backend.features.build import FEATURES, build_features
from backend.models.market_index import fit_trend, monthly_index, trend_value

log = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, random_state=42, verbose=-1,
)
INTERVAL_LEVEL = (0.10, 0.90)  # 80 % интервал
HORIZON_WIDEN = 0.01           # эвристика: +1 % к ширине интервала за каждый месяц горизонта


def _fit_model(df: pd.DataFrame, idx: pd.Series, params: dict) -> tuple[LGBMRegressor, pd.DataFrame]:
    d = df[df["month"].isin(idx.index)]
    y = np.log(d["price_m2"]) - d["month"].map(idx)
    model = LGBMRegressor(**params)
    model.fit(build_features(d), y)
    return model, d


def _hedonic_residuals(model, df: pd.DataFrame, idx_all: pd.Series) -> np.ndarray:
    """Остатки модели в логах при известном индексе месяца (шум без ошибки тренда)."""
    d = df[df["month"].isin(idx_all.index)]
    y = np.log(d["price_m2"]) - d["month"].map(idx_all)
    return (y - model.predict(build_features(d))).to_numpy()


def train_pipeline(df: pd.DataFrame, test_months: int = 3, calib_months: int = 3,
                   params: dict | None = None) -> tuple[dict, dict]:
    """Обучить и оценить. Возвращает (bundle для сохранения, метрики)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    last = int(df["month"].max())
    test_start = last - test_months + 1
    calib_start = test_start - calib_months

    train = df[df["month"] < calib_start]
    calib = df[(df["month"] >= calib_start) & (df["month"] < test_start)]
    test = df[df["month"] >= test_start]
    for name, part in (("train", train), ("calib", calib), ("test", test)):
        if len(part) < 100:
            raise DataValidationError(f"Слишком мало данных в части {name}: {len(part)} строк")

    # --- Оценка по времени ---
    idx_train = monthly_index(train)
    trend_train = fit_trend(idx_train)
    train_last = int(idx_train.index.max())
    model, _ = _fit_model(train, idx_train, params)

    idx_all_months = monthly_index(df)
    q_low, q_high = np.quantile(_hedonic_residuals(model, calib, idx_all_months), INTERVAL_LEVEL)

    X_test = build_features(test)
    m = test["month"].to_numpy()
    pred_log = model.predict(X_test) + trend_value(trend_train, m)
    pred_price = np.exp(pred_log) * test["area"].to_numpy()
    true_price = test["price"].to_numpy()

    h = np.maximum(m - train_last, 1)
    lo = np.exp(pred_log + q_low - HORIZON_WIDEN * h) * test["area"].to_numpy()
    hi = np.exp(pred_log + q_high + HORIZON_WIDEN * h) * test["area"].to_numpy()

    base_m2 = float(np.median(train["price_m2"]))
    base_price = base_m2 * test["area"].to_numpy()

    def mape(p):
        return float(np.mean(np.abs(p - true_price) / true_price))

    metrics = {
        "rows_train": len(train), "rows_calib": len(calib), "rows_test": len(test),
        "test_months": f"{int(m.min())}..{int(m.max())}",
        "mape_model": mape(pred_price),
        "mape_baseline_median": mape(base_price),
        "mae_rub": float(np.mean(np.abs(pred_price - true_price))),
        "rmse_rub": float(np.sqrt(np.mean((pred_price - true_price) ** 2))),
        "interval_coverage_80": float(np.mean((true_price >= lo) & (true_price <= hi))),
    }

    # --- Финальная модель на всех данных ---
    idx_final = monthly_index(df)
    trend_final = fit_trend(idx_final)
    final_model, used = _fit_model(df, idx_final, params)
    bundle = {
        "model": final_model, "trend": trend_final,
        "last_month": int(idx_final.index.max()),
        "q_low": float(q_low), "q_high": float(q_high),
        "features": FEATURES,
        "geo_median": (float(df["geo_lat"].median()), float(df["geo_lon"].median())),
        "rows": len(used), "metrics": metrics,
    }
    return bundle, metrics


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)
