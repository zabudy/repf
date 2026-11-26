"""Таблицы: объект недвижимости, прогноз, обратная связь эксперта."""
from datetime import UTC, date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
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
