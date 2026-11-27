"""Allow-listed forecasting methods with strict, bounded argument schemas."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pydantic import Field

from backend.core.forecast import InputModel, TaskType, ToolChoice
from backend.timeseries import classical


class ARArguments(InputModel):
    lags: int = Field(default=12, ge=1, le=64, strict=True)
    ridge: float = Field(default=1.0, ge=.001, le=1000, strict=True)
    differences: int = Field(default=1, ge=0, le=1, strict=True)


class HoltArguments(InputModel):
    alpha: float = Field(default=.4, ge=.01, le=1, strict=True)
    beta: float = Field(default=.12, ge=.01, le=1, strict=True)
    damping: float = Field(default=.92, ge=.6, le=1, strict=True)


class HarmonicArguments(InputModel):
    harmonics: int = Field(default=3, ge=1, le=12, strict=True)


class RidgeArguments(InputModel):
    alpha: float = Field(default=8, ge=.001, le=10000, strict=True)


class NeighborsArguments(InputModel):
    neighbors: int = Field(default=25, ge=2, le=100, strict=True)


class BoostArguments(InputModel):
    rounds: int = Field(default=32, ge=5, le=100, strict=True)


class EnsembleArguments(InputModel):
    ridge_weight: float = Field(default=.7, ge=0, le=1, strict=True)


@dataclass(frozen=True)
class Tool:
    task_type: TaskType
    description: str
    schema: type[InputModel]
    run: Callable | None = None

    def definition(self, name: str) -> dict:
        parameters = self.schema.model_json_schema()
        parameters.pop("title", None)
        return {"type": "function", "function": {
            "name": name, "description": self.description, "parameters": parameters,
        }}


REGISTRY = {
    "fit_holt": Tool("timeseries", "Damped Holt exponential smoothing for trend", HoltArguments,
                    lambda y, h, p, **kw: classical.holt(y, h, **kw)),
    "fit_autoregression": Tool("timeseries", "Regularized AR on levels or differences for autocorrelation",
                              ARArguments, lambda y, h, p, **kw: classical.fit_autoregression(y, h, **kw)),
    "fit_harmonic": Tool("timeseries", "Trend and Fourier seasonality for periodic data", HarmonicArguments,
                        lambda y, h, p, **kw: classical.harmonic(y, h, p, **kw)),
    "fit_ridge": Tool("tabular", "Regularized hedonic regression on one-hot features", RidgeArguments),
    "fit_neighbors": Tool("tabular", "Distance-weighted regression using comparable apartments",
                          NeighborsArguments),
    "fit_boost": Tool("tabular", "Gradient boosting on regression stumps", BoostArguments),
    "fit_ensemble": Tool("tabular", "Weighted ensemble of Ridge, neighbors and boosted stumps",
                         EnsembleArguments),
}


def validate(choice: ToolChoice, task_type: TaskType) -> ToolChoice:
    tool = REGISTRY.get(choice.name)
    if tool is None or tool.task_type != task_type:
        raise ValueError("Инструмент не зарегистрирован для этого типа задачи")
    return ToolChoice(name=choice.name, arguments=tool.schema.model_validate(choice.arguments).model_dump())


def available(task_type: TaskType, attempted: list[str] | None = None) -> list[str]:
    return [name for name, tool in REGISTRY.items()
            if tool.task_type == task_type and name not in (attempted or [])]


def series_prediction(choice: ToolChoice, y: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if choice.name == "seasonal_naive":
        return classical.seasonal_naive(y, horizon, period)
    tool = REGISTRY[choice.name]
    if tool.run is None:
        raise ValueError("Инструмент не поддерживает временные ряды")
    return tool.run(y, horizon, period, **choice.arguments)
