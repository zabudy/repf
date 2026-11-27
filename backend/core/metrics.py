"""Evaluation measures and holdout residual diagnostics."""
import numpy as np

from backend.timeseries.preprocessing import acf


def accuracy(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray, period: int) -> dict:
    errors = np.abs(actual - forecast)
    nonzero = np.abs(actual) > 1e-8
    lag = period if len(history) > period else 1
    scale = np.mean(np.abs(history[lag:] - history[:-lag])) if len(history) > lag else 0
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(errors[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else None,
        "mase": float(np.mean(errors) / scale) if scale > 1e-8 else None,
        "wape": float(np.sum(errors) / np.sum(np.abs(actual)) * 100) if nonzero.any() else None,
    }


def residual_diagnostics(residuals: np.ndarray, statistical: bool = True) -> dict:
    n = len(residuals)
    lags = min(10, n // 3)
    stats: dict = {"n": n, "acf": [acf(residuals, i) for i in range(1, lags + 1)]}
    if not statistical or n < 4 or np.std(residuals) < 1e-10:
        return stats
    try:
        from scipy.stats import shapiro
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import pacf
    except ImportError:
        return stats
    stats["pacf"] = pacf(residuals, nlags=lags, method="ywm")[1:].tolist()
    stats["ljung_box_pvalue"] = float(acorr_ljungbox(residuals, lags=[lags]).iloc[0]["lb_pvalue"])
    stats["shapiro_pvalue"] = float(shapiro(residuals[:5000]).pvalue)
    return stats
