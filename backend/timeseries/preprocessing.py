"""Normalisation, feature extraction and temporal holdouts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.core.forecast import SeriesInput

SEASONS = {"D": 7, "W": 52, "MS": 12, "QS": 4}


@dataclass(frozen=True)
class Series:
    index: pd.DatetimeIndex
    values: np.ndarray
    frequency: str
    period: int
    nonnegative: bool
    missing_rate: float


def prepare(data: SeriesInput) -> Series:
    frame = pd.DataFrame([row.model_dump() for row in data.observations])
    frame["date"] = pd.to_datetime(frame["date"])
    raw = frame.groupby("date")["value"].mean().sort_index().astype(float)
    if len(raw) < 12:
        raise ValueError("Нужны минимум 12 различных дат")
    inferred = pd.infer_freq(raw.index)
    aliases = {"QS-OCT": "QS", "QS-JAN": "QS", "QE-DEC": "QS", "ME": "MS"}
    frequency = data.frequency or aliases.get(inferred, inferred)
    if frequency is None:
        delta = float(np.median(np.diff(raw.index.values) / np.timedelta64(1, "D")))
        frequency = "QS" if delta >= 80 else "MS" if delta >= 25 else "W" if delta >= 6 else "D"
    if frequency.startswith("W-"):
        frequency = "W"
    if frequency not in SEASONS:
        raise ValueError("Укажите частоту D, W, MS или QS")
    bins = raw.resample(frequency)
    if bins.ngroups > 25000:
        raise ValueError("Интервал между датами слишком велик для заданной частоты")
    values = bins.mean().dropna()
    if len(values) < 12:
        raise ValueError("После ресемплинга нужно минимум 12 непустых периодов")
    regular = values.asfreq(frequency)
    missing_rate = float(regular.isna().mean())
    if missing_rate > 0.50:
        raise ValueError("Более половины периодов пропущено")
    # A forward fill is causal even if a gap crosses a validation cutoff.
    values = regular.ffill().to_numpy(dtype=float)
    if data.nonnegative and np.any(values < 0):
        raise ValueError("В неотрицательном ряду найдены значения ниже нуля")
    return Series(regular.index, values, frequency, data.seasonal_period or SEASONS[frequency],
                  data.nonnegative, missing_rate)


def acf(values: np.ndarray, lag: int) -> float:
    if len(values) < lag + 2:
        return 0.0
    left, right = values[:-lag], values[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1, 1))


def feature_vector(values: np.ndarray) -> np.ndarray:
    """Scale-free lag, distribution and spectral features (>256 raw dimensions)."""
    scale = max(float(np.std(values)), 1e-8)
    z = (values - values.mean()) / scale
    positions = np.linspace(0, len(z) - 1, 128)
    levels = np.interp(positions, np.arange(len(z)), z)
    differences = np.diff(z)
    histogram, _ = np.histogram(z, bins=12)
    probability = histogram[histogram > 0] / len(z)
    spectrum = np.abs(np.fft.rfft(levels))[1:]
    spectrum = spectrum / max(float(spectrum.sum()), 1e-8)
    return np.nan_to_num(np.r_[
        np.log1p(len(values)), np.mean(differences), np.std(differences),
        -np.sum(probability * np.log2(probability)),
        [acf(z, lag) for lag in range(1, 65)],
        np.quantile(z, np.linspace(0, 1, 24)),
        levels, spectrum,
    ])


def diagnostics(values: np.ndarray, period: int, statistical: bool = True) -> dict:
    scale = max(float(values.std()), 1e-8)
    trend = float((np.mean(values[-min(6, len(values)):]) -
                   np.mean(values[:min(6, len(values))])) / scale)
    output = {
        "count": len(values), "trend": trend, "seasonality": acf(values, period),
        "autocorrelation": acf(values, 1), "volatility": float(np.std(np.diff(values)) / scale),
    }
    if not statistical:
        return output
    try:
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        output["statistical_backend"] = "unavailable"
        return output
    if scale > 1e-8:
        output["adf_pvalue"] = float(adfuller(values, maxlag=min(8, len(values) // 4))[1])
        if len(values) >= 2 * period:
            stl = STL(values, period=period, robust=True).fit()
            output["stl_strength"] = float(max(0, 1 - np.var(stl.resid) /
                                               max(np.var(stl.seasonal + stl.resid), 1e-12)))
    return output
