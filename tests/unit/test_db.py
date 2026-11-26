import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db import models
from backend.db.base import Base


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()


async def test_property_with_forecast_and_feedback(session):
    p = models.Property(region="Kazan", area_total=40.0, floor=2, floors_total=9)
    f = models.Forecast(horizon_months=3, predicted_price=5_000_000.0, confidence=0.5)
    f.feedback.append(models.ExpertFeedback(expert_price=5_200_000.0, comment="ok"))
    p.forecasts.append(f)
    session.add(p)
    await session.commit()

    got = (await session.execute(select(models.Property))).scalar_one()
    assert got.id == 1
    assert got.created_at is not None
    assert len(got.forecasts) == 1
    assert got.forecasts[0].needs_review is False


async def test_cascade_delete(session):
    p = models.Property(region="Kazan", area_total=40.0, floor=1, floors_total=5)
    p.forecasts.append(models.Forecast(horizon_months=1, predicted_price=1.0))
    session.add(p)
    await session.commit()
    await session.delete(p)
    await session.commit()
    assert (await session.execute(select(models.Forecast))).first() is None
