"""Register and sign in with an email address."""
import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.api.deps import CurrentUser, Session
from backend.core.config import Settings, get_settings
from backend.core.forecast import InputModel
from backend.db.models import User
from backend.db.session import SessionLocal
from backend.utils.security import hash_password, issue_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(InputModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def size(cls, value: str) -> str:
        if len(value.encode()) > 72:
            raise ValueError("Пароль должен занимать не более 72 байт")
        return value


def summary(user: User) -> dict:
    return {"id": user.id, "email": user.email, "role": user.role}


def _signed(user: User) -> dict:
    return {"access_token": issue_token(user.id, get_settings()), "token_type": "bearer", "user": summary(user)}


@router.post("/register", status_code=201)
async def register(payload: Credentials, session: Session) -> dict:
    user = User(id=str(uuid4()), email=payload.email.casefold(),
                password_hash=await asyncio.to_thread(hash_password, payload.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        raise HTTPException(409, "Этот адрес уже зарегистрирован") from error
    return _signed(user)


@router.post("/login")
async def login(payload: Credentials, session: Session) -> dict:
    user = await session.scalar(select(User).where(User.email == payload.email.casefold()))
    if user is None or not await asyncio.to_thread(verify_password, payload.password, user.password_hash):
        raise HTTPException(401, "Неверный адрес или пароль")
    return _signed(user)


@router.get("/me")
async def me(user: CurrentUser) -> dict:
    return summary(user)


async def ensure_admin(settings: Settings) -> None:
    if not settings.ADMIN_EMAIL or not settings.ADMIN_PASSWORD:
        return
    credentials = Credentials(email=settings.ADMIN_EMAIL, password=settings.ADMIN_PASSWORD.get_secret_value())
    email = credentials.email.casefold()
    async with SessionLocal.begin() as session:
        current = await session.scalar(select(User).where(User.email == email))
        if current is None:
            session.add(User(id=str(uuid4()), email=email, role="admin",
                             password_hash=await asyncio.to_thread(hash_password, credentials.password)))
