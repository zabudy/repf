"""Deterministic classical forecasters on equally spaced observations."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def seasonal_naive(y: np.ndarray, horizon: int, period: int) -> np.ndarray:
    season = period if len(y) >= period else 1
    return np.resize(y[-season:], horizon)


def holt(y: np.ndarray, horizon: int, alpha: float, beta: float, damping: float) -> np.ndarray:
    level, trend = float(y[0]), float(np.median(np.diff(y[:min(8, len(y))])))
    for observed in y[1:]:
        previous_level = level
        level = alpha * observed + (1 - alpha) * (level + damping * trend)
        trend = beta * (level - previous_level) + (1 - beta) * damping * trend
    return level + trend * np.cumsum(damping ** np.arange(1, horizon + 1))


def fit_autoregression(y: np.ndarray, horizon: int, lags: int, ridge: float,
                       differences: int) -> np.ndarray:
    """AR(p) on levels or first differences, with ridge regularisation."""
    transformed = np.diff(y) if differences else y
    lags = min(lags, max(1, (len(transformed) - 1) // 3))
    windows = sliding_window_view(transformed, lags + 1)
    mean, scale = float(transformed.mean()), max(float(transformed.std()), 1e-8)
    normal = (windows - mean) / scale
    x = np.c_[np.ones(len(normal)), normal[:, :-1]]
    penalty = np.diag([0] + [ridge] * lags)
    weights = np.linalg.lstsq(x.T @ x + penalty, x.T @ normal[:, -1], rcond=None)[0]
    history = list((transformed[-lags:] - mean) / scale)
    forecast = []
    for _ in range(horizon):
        prediction = float(np.clip(weights[0] + history[-lags:] @ weights[1:], -8, 8))
        forecast.append(prediction * scale + mean)
        history.append(prediction)
    result = np.asarray(forecast)
    return y[-1] + np.cumsum(result) if differences else result


def harmonic(y: np.ndarray, horizon: int, period: int, harmonics: int) -> np.ndarray:
    """Trend plus Fourier seasonality estimated by ridge regression."""
    n = len(y)
    harmonics = min(harmonics, (period - 1) // 2) if n >= period * 2 else 0
    t = np.arange(n + horizon)
    columns = [np.ones(len(t)), t / n]
    for k in range(1, harmonics + 1):
        angle = t * (2 * np.pi * k / period)
        columns.extend([np.sin(angle), np.cos(angle)])
    matrix = np.column_stack(columns)
    training = matrix[:n]
    weights = np.linalg.lstsq(training.T @ training + np.diag([0, .1] + [.2] * (harmonics * 2)),
                             training.T @ y, rcond=None)[0]
    return matrix[n:] @ weights
