"""Структуры данных, которыми обмениваются агенты."""
from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    region: str = "Kazan"
    district: str | None = None
    area_total: float
    floor: int
    floors_total: int
    build_year: int | None = None
    house_type: str | None = None
    condition: str | None = None
    horizon_months: int = 6
    # необязательные поля, которые использует ML-модель
    geo_lat: float | None = None
    geo_lon: float | None = None
    rooms: int | None = None  # 0 = студия
    kitchen_area: float | None = None


class ValidationReport(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)    # исправить нельзя — прогноза не будет
    warnings: list[str] = Field(default_factory=list)  # прогноз строим, но уверенность ниже


class RawForecast(BaseModel):
    price: float
    interval_low: float
    interval_high: float
    model_name: str
    notes: list[str] = Field(default_factory=list)  # ограничения модели


class Critique(BaseModel):
    confidence: float  # 0..1
    notes: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    agent: str
    status: str  # ok | warning | error
    message: str


class ForecastResult(BaseModel):
    price: float | None = None
    price_per_m2: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    confidence: float | None = None
    needs_review: bool = False
    model_name: str | None = None
    explanation: str = ""
    trace: list[AgentStep] = Field(default_factory=list)
