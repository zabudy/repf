"""Deterministic checks stay authoritative when the LLM evaluates a forecast."""
import numpy as np

from backend.agents.base import Agent, Artifacts, Trace
from backend.core.forecast import ForecastState


class Validator(Agent):
    def __init__(self, artifacts: Artifacts, trace: Trace):
        self.artifacts, self.trace = artifacts, trace

    async def process(self, state: ForecastState, parent_id: str) -> None:
        async with self.trace.step("Validator", parent_id, horizon=state.horizon) as (_, output):
            forecast = self.artifacts.get(state.result_id)
            y = np.asarray(forecast["predictions"])
            prepared, _ = self.artifacts.get(state.prepared_id)
            nonnegative = state.task_type == "tabular" or prepared.nonnegative
            last_significance = forecast["residuals"].get("ljung_box_pvalue")
            reasons = []
            if len(y) != state.horizon or len(forecast["dates"]) != state.horizon:
                reasons.append("Неверная длина прогноза")
            if not np.isfinite(y).all() or (nonnegative and (y < 0).any()):
                reasons.append("Недопустимые значения в прогнозе")
            if state.task_type == "tabular":
                if state.metadata["district_comparables"] < 5:
                    reasons.append("Мало аналогов в указанном районе")
                if forecast["metrics"]["wape"] is not None and forecast["metrics"]["wape"] > 35:
                    reasons.append("WAPE оценки квартиры превышает 35%")
            elif last_significance is not None and last_significance < .01:
                reasons.append("На отложенных остатках обнаружена автокорреляция")
            if state.task_type == "timeseries":
                mase = forecast["metrics"]["mase"]
                if mase is not None and mase > 1.6:
                    reasons.append("Ошибка выше сезонного наивного ориентира по MASE")
            state.validation_passed = not reasons
            state.diagnostics = {"reasons": reasons, "valid": state.validation_passed,
                                 "residuals": forecast["residuals"]}
            output.update(state.diagnostics)
