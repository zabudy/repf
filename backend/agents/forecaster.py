"""Execute one validated model, measure temporal holdout error and calibrate intervals."""
import asyncio
import math

import numpy as np
import pandas as pd

from backend.agents.base import Agent, Artifacts, Trace
from backend.core.forecast import ForecastState, ToolChoice
from backend.core.llm.tools import series_prediction
from backend.core.metrics import accuracy, residual_diagnostics
from backend.tabular import housing
from backend.timeseries.preprocessing import Series


def forecast_series(choice: ToolChoice, series: Series, history: np.ndarray, horizon: int) -> np.ndarray:
    prediction = series_prediction(choice, history, horizon, series.period)
    return np.maximum(prediction, 0) if series.nonnegative else prediction


def run_forecast(prepared, state: ForecastState, statistical: bool = True) -> dict:
    choice, horizon = state.choice, state.horizon
    if choice is None:
        raise ValueError("Модель не выбрана")
    if isinstance(prepared, Series):
        y = prepared.values
        width = min(horizon, max(3, len(y) // 5))
        end = len(y) - width
        # Three non-overlapping holdouts when there is enough history.
        positions = [pos for pos in (end - 2 * width, end - width, end) if pos >= 8]
        holdouts = [forecast_series(choice, prepared, y[:pos], width) for pos in positions]
        actual = np.concatenate([y[pos:pos + width] for pos in positions])
        fitted = np.concatenate(holdouts)
        history = y[:positions[0]]
        forecast = forecast_series(choice, prepared, y, horizon)
        result = {
            "predictions": forecast.tolist(), "frequency": prepared.frequency,
            "dates": pd.date_range(prepared.index[-1], periods=horizon + 1,
                                   freq=prepared.frequency)[1:].strftime("%Y-%m-%d").tolist(),
        }
        period = prepared.period
    else:
        actual, fitted, history = housing.backtest(prepared, choice.name, choice.arguments)
        result = housing.forecast(prepared, horizon, choice.name, choice.arguments)
        forecast = np.asarray(result["predictions"])
        period = 1
    metrics = accuracy(actual, fitted, history, period)
    errors = np.abs(actual - fitted)
    quantile = min(1., math.ceil((len(errors) + 1) * .8) / len(errors))
    radius = float(np.quantile(errors, quantile, method="higher"))
    if not isinstance(prepared, Series) or prepared.nonnegative:
        lower = np.maximum(forecast - radius, 0)
    else:
        lower = forecast - radius
    return {**result, "model": choice.model_dump(), "metrics": metrics,
            "residuals": residual_diagnostics(actual - fitted, statistical),
            "interval": {"level": .8, "lower": lower.tolist(), "upper": (forecast + radius).tolist(),
                         "calibration_n": len(errors), "calibration_coverage": float(np.mean(errors <= radius))}}


class Forecaster(Agent):
    def __init__(self, artifacts: Artifacts, trace: Trace, statistical: bool):
        self.artifacts, self.trace, self.statistical = artifacts, trace, statistical

    async def process(self, state: ForecastState, parent_id: str) -> None:
        assert state.choice is not None
        async with self.trace.step("Forecaster", parent_id, tool=state.choice.model_dump()) as (_, output):
            prepared, _ = self.artifacts.get(state.prepared_id)
            result = await asyncio.to_thread(run_forecast, prepared, state, self.statistical)
            state.result_id = self.artifacts.put(result)
            output.update(metrics=result["metrics"], forecast_points=len(result["predictions"]))
