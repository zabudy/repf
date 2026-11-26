"""Подключение к базе данных и выдача сессий."""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import get_settings

engine = create_async_engine(get_settings().DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Зависимость для FastAPI (пригодится на этапе 9)."""
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Создать все таблицы, если их ещё нет."""
    from backend.db import models  # noqa: F401  (регистрирует таблицы)
    from backend.db.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
