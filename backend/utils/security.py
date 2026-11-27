"""Password storage and signed, expiring sessions."""
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import scrypt
from hmac import compare_digest
from secrets import token_bytes, token_urlsafe
from datetime import UTC, datetime, timedelta

try:
    from jose import jwt
    from jose.exceptions import JWTError
except ImportError:  # PyJWT can also be used in a minimal developer environment.
    import jwt
    from jwt import PyJWTError as JWTError

from backend.core.config import Settings

LOCAL_SECRET = token_urlsafe(48)


def hash_password(password: str) -> str:
    raw = password.encode()
    try:
        import bcrypt
    except ImportError:
        salt = token_bytes(16)
        output = scrypt(raw, salt=salt, n=2 ** 14, r=8, p=1)
        return "scrypt$" + urlsafe_b64encode(salt + output).decode()
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode()


def verify_password(password: str, stored: str) -> bool:
    raw = password.encode()
    if stored.startswith("scrypt$"):
        value = urlsafe_b64decode(stored.split("$", 1)[1])
        actual = scrypt(raw, salt=value[:16], n=2 ** 14, r=8, p=1)
        return compare_digest(actual, value[16:])
    try:
        import bcrypt
        return bcrypt.checkpw(raw, stored.encode())
    except (ImportError, ValueError):
        return False


def _secret(settings: Settings) -> str:
    return settings.JWT_SECRET_KEY.get_secret_value() if settings.JWT_SECRET_KEY else LOCAL_SECRET


def issue_token(user_id: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)},
                      _secret(settings), algorithm="HS256")


def token_user(token: str, settings: Settings) -> str | None:
    try:
        payload = jwt.decode(token, _secret(settings), algorithms=["HS256"])
        if "exp" in payload and isinstance(payload.get("sub"), str):
            return payload["sub"]
    except JWTError:
        pass
    return None
