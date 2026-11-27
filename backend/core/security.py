"""Пароли (bcrypt) и токены (JWT)."""
from datetime import UTC, datetime, timedelta

import bcrypt
from jose import JWTError, jwt

from backend.core.config import get_settings

MAX_PASSWORD_BYTES = 72  # ограничение bcrypt


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError("Пароль слишком длинный (максимум 72 байта)")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    s = get_settings()
    expire = datetime.now(UTC) + timedelta(minutes=s.JWT_EXPIRATION_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": expire}, s.JWT_SECRET_KEY, algorithm=s.JWT_ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """Вернуть id пользователя из токена или None, если токен плохой/просрочен."""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.JWT_SECRET_KEY, algorithms=[s.JWT_ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
