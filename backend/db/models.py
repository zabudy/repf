"""Таблицы: объект недвижимости, прогноз, обратная связь эксперта."""
from datetime import UTC, date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Property(Base):
    """Квартира: параметры и (если известна) фактическая цена."""

    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(primary_key=True)
    region: Mapped[str] = mapped_column(String(64), index=True)
    district: Mapped[str | None] = mapped_column(String(128), default=None)
    address: Mapped[str | None] = mapped_column(String(256), default=None)
    area_total: Mapped[float] = mapped_column(Float)  # м²
    floor: Mapped[int] = mapped_column(Integer)
    floors_total: Mapped[int] = mapped_column(Integer)
    build_year: Mapped[int | None] = mapped_column(Integer, default=None)
    house_type: Mapped[str | None] = mapped_column(String(32), default=None)
    condition: Mapped[str | None] = mapped_column(String(32), default=None)
    price_rub: Mapped[float | None] = mapped_column(Float, default=None)  # фактическая цена
    source: Mapped[str | None] = mapped_column(String(64), default=None)
    listing_date: Mapped[date | None] = mapped_column(Date, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="property", cascade="all, delete-orphan"
    )


class Forecast(Base):
    """Результат прогноза по объекту."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True)
    property_id: Mapped[int] = mapped_column(ForeignKey("properties.id"), index=True)
    horizon_months: Mapped[int] = mapped_column(Integer)
    predicted_price: Mapped[float] = mapped_column(Float)
    interval_low: Mapped[float | None] = mapped_column(Float, default=None)
    interval_high: Mapped[float | None] = mapped_column(Float, default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)  # 0..1
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    model_name: Mapped[str | None] = mapped_column(String(64), default=None)
    explanation: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    property: Mapped[Property] = relationship(back_populates="forecasts")
    feedback: Mapped[list["ExpertFeedback"]] = relationship(
        back_populates="forecast", cascade="all, delete-orphan"
    )


class ExpertFeedback(Base):
    """Поправка эксперта к прогнозу (для Human-in-the-Loop)."""

    __tablename__ = "expert_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    forecast_id: Mapped[int] = mapped_column(ForeignKey("forecasts.id"), index=True)
    expert_price: Mapped[float] = mapped_column(Float)
    comment: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    forecast: Mapped[Forecast] = relationship(back_populates="feedback")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ForecastTask(Base):
    __tablename__ = "forecast_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    request: Mapped[dict] = mapped_column(JSON)
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    review: Mapped[dict | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ReasoningLog(Base):
    __tablename__ = "reasoning_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("forecast_tasks.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("reasoning_logs.id"), default=None)
    agent: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(16))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("forecast_tasks.id"), index=True)
    span_id: Mapped[str] = mapped_column(ForeignKey("reasoning_logs.id"))
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KnowledgeCase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(24), index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class GlossaryTerm(Base):
    __tablename__ = "glossary"

    term: Mapped[str] = mapped_column(String(200), primary_key=True)
    definition: Mapped[str] = mapped_column(Text)


class EvaluationRun(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    parameters: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CacheEntry(Base):
    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LLMProfile(Base):
    __tablename__ = "llm_profiles"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    provider: Mapped[str] = mapped_column(String(24))
    model: Mapped[str | None] = mapped_column(String(160))
    endpoint: Mapped[str | None] = mapped_column(String(256))
    encrypted_key: Mapped[str | None] = mapped_column(Text)


class LLMCheckpoint(Base):
    __tablename__ = "llm_checkpoints"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    provider: Mapped[str] = mapped_column(String(24))
    model: Mapped[str | None] = mapped_column(String(160))
    endpoint: Mapped[str | None] = mapped_column(String(256))
    encrypted_key: Mapped[str | None] = mapped_column(Text)


class LLMSelection(Base):
    __tablename__ = "llm_selections"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    checkpoint_id: Mapped[str | None] = mapped_column(ForeignKey("llm_checkpoints.id", ondelete="SET NULL"))
