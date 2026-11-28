"""Schemas shared by the API, forecast worker and evaluation runner."""
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

TaskType = Literal["timeseries", "tabular"]
Mode = Literal["baseline", "+rag", "+agents"]
Status = Literal["pending", "running", "waiting_for_human", "completed", "failed"]
Positive = Annotated[FiniteFloat, Field(gt=0)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(InputModel):
    date: date
    value: FiniteFloat | None


class SeriesInput(InputModel):
    task_type: Literal["timeseries"] = "timeseries"
    observations: list[Observation] = Field(min_length=12, max_length=10000)
    frequency: Literal["D", "W", "MS", "QS"] | None = None
    seasonal_period: int | None = Field(default=None, ge=2, le=366)
    nonnegative: bool = False


class Apartment(InputModel):
    district: str = Field(min_length=1, max_length=60)
    area_total: Positive = Field(le=1000)
    floor: int | None = Field(default=None, ge=1, le=150)
    floors_total: int | None = Field(default=None, ge=1, le=150)
    build_year: int | None = Field(default=None, ge=1850, le=2100)
    house_type: str | None = Field(default=None, max_length=40)
    condition: str | None = Field(default=None, max_length=40)

    @model_validator(mode="after")
    def check_floor(self) -> "Apartment":
        if self.floor is not None and self.floors_total is not None and self.floor > self.floors_total:
            raise ValueError("Этаж выше этажности дома")
        return self


class Sale(Apartment):
    date: date
    price_rub: Positive


class TabularInput(InputModel):
    task_type: Literal["tabular"] = "tabular"
    apartment: Apartment
    market: list[Sale] | None = Field(default=None, min_length=40, max_length=20000)
    market_index: SeriesInput | None = None
    target_column: Literal["price_rub"] = "price_rub"


ForecastInput = Annotated[SeriesInput | TabularInput, Field(discriminator="task_type")]


class ForecastRequest(InputModel):
    data: ForecastInput
    horizon: int = Field(default=6, ge=1, le=366)
    mode: Mode = "+agents"

    @model_validator(mode="after")
    def check_horizon(self) -> "ForecastRequest":
        if self.data.task_type == "tabular" and self.horizon > 12:
            raise ValueError("Для оценки квартиры горизонт составляет 1–12 месяцев")
        return self


class ToolChoice(InputModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    arguments: dict[str, int | float] = Field(default_factory=dict)


class ForecastState(BaseModel):
    task_id: str
    input_id: str
    task_type: TaskType
    horizon: int
    mode: Mode
    target_column: str | None = None
    prepared_id: str | None = None
    result_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    retrieved: list[dict] = Field(default_factory=list)
    choice: ToolChoice | None = None
    attempted: list[str] = Field(default_factory=list)
    iteration: int = 0
    validation_passed: bool = False
    critic_score: float | None = None
    critic_comment: str | None = None
    hitl_resolution: dict | None = None


class Review(InputModel):
    action: Literal["accept", "retry", "reject"]
    model: str | None = None
    horizon: int | None = Field(default=None, ge=1, le=366)
    comment: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def check_action(self) -> "Review":
        if self.action == "retry" and self.model is None and self.horizon is None:
            raise ValueError("Для повтора укажите модель или горизонт")
        if self.action != "retry" and (self.model is not None or self.horizon is not None):
            raise ValueError("Модель и горизонт можно менять только при повторе")
        return self
