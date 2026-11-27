"""Индекс рынка: средний лог цены за м² по месяцам + простой линейный тренд."""
import numpy as np
import pandas as pd


def monthly_index(df: pd.DataFrame, min_rows: int = 20) -> pd.Series:
    """Индекс по месяцам: среднее ln(цена за м²). Месяцы с малым числом объявлений отбрасываются."""
    g = np.log(df["price_m2"]).groupby(df["month"])
    idx = g.mean()[g.size() >= min_rows]
    return idx.sort_index()


def fit_trend(idx: pd.Series, window: int = 12) -> tuple[float, float]:
    """Прямая по последним `window` месяцам: ln(цена/м²) ≈ slope * month + intercept."""
    last = idx.iloc[-window:]
    if len(last) < 3:
        raise ValueError("Слишком мало месяцев для оценки тренда")
    slope, intercept = np.polyfit(last.index.to_numpy(dtype=float), last.to_numpy(), 1)
    return float(slope), float(intercept)


def trend_value(trend: tuple[float, float], month) -> np.ndarray | float:
    slope, intercept = trend
    return slope * np.asarray(month, dtype=float) + intercept
