"""Bounded-lifetime SQL cache shared by workers."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

from sqlalchemy import delete, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.db.models import CacheEntry
from backend.db.session import SessionLocal, engine


def cache_key(namespace: str, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return namespace + ":" + sha256(data.encode()).hexdigest()


class DatabaseCache:
    async def get(self, key: str) -> dict | None:
        async with SessionLocal.begin() as session:
            result = await session.execute(update(CacheEntry).where(
                CacheEntry.key == key, CacheEntry.expires_at > datetime.now(UTC),
            ).values(hits=CacheEntry.hits + 1).returning(CacheEntry.payload))
            return result.scalar_one_or_none()

    async def put(self, key: str, payload: dict) -> None:
        insert = sqlite_insert if engine.dialect.name == "sqlite" else postgres_insert
        query = insert(CacheEntry).values(key=key, payload=payload, expires_at=datetime.now(UTC) + timedelta(hours=12))
        async with SessionLocal.begin() as session:
            await session.execute(query.on_conflict_do_update(
                index_elements=["key"], set_={"payload": query.excluded.payload,
                                               "expires_at": query.excluded.expires_at, "hits": 0},
            ))

    async def prune(self) -> None:
        async with SessionLocal.begin() as session:
            await session.execute(delete(CacheEntry).where(CacheEntry.expires_at <= datetime.now(UTC)))
