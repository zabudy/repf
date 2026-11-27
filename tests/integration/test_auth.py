import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.api.main import app
from backend.db import models  # noqa: F401
from backend.db.base import Base
from backend.db.session import get_session


@pytest.fixture
async def client():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


USER = {"email": "Ivan@Mail.com", "password": "secret123", "full_name": "Иван"}


async def test_register_login_me(client):
    r = await client.post("/auth/register", json=USER)
    assert r.status_code == 201
    assert r.json()["email"] == "ivan@mail.com"
    assert "password" not in r.text and "hashed" not in r.text

    r = await client.post("/auth/login", data={"username": "IVAN@mail.com", "password": "secret123"})
    assert r.status_code == 200
    token = r.json()["access_token"]

    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["role"] == "user"


async def test_duplicate_email(client):
    await client.post("/auth/register", json=USER)
    r = await client.post("/auth/register", json=USER)
    assert r.status_code == 409


async def test_wrong_password(client):
    await client.post("/auth/register", json=USER)
    r = await client.post("/auth/login", data={"username": USER["email"], "password": "nope"})
    assert r.status_code == 401


async def test_me_without_token(client):
    assert (await client.get("/auth/me")).status_code == 401


async def test_short_password(client):
    r = await client.post("/auth/register", json={**USER, "password": "123"})
    assert r.status_code == 422
