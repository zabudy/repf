"""User-owned provider configuration and encrypted API keys."""
from base64 import urlsafe_b64encode
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from pydantic import Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings
from backend.core.forecast import InputModel
from backend.db.models import LLMCheckpoint, LLMProfile, LLMSelection

KEY_PATH = Path(__file__).resolve().parents[3] / ".repf-llm.key"
ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Groq": "https://api.groq.com/openai/v1",
}


class ProfileInput(InputModel):
    provider: Literal["deepseek", "compatible", "ollama", "local"]
    model: str | None = Field(default=None, max_length=160)
    endpoint: str | None = Field(default=None, max_length=256)
    api_key: SecretStr | None = None


def endpoints(settings: Settings) -> dict[str, str]:
    result = ENDPOINTS.copy()
    if settings.COMPATIBLE_LLM_URL:
        result["Сервер из .env"] = settings.COMPATIBLE_LLM_URL.rstrip("/")
    return result


def _cipher(settings: Settings) -> Fernet:
    secret = settings.JWT_SECRET_KEY
    if secret and len(secret.get_secret_value()) >= 32:
        material = ("repf:llm-profile:v1:" + secret.get_secret_value()).encode()
        return Fernet(urlsafe_b64encode(sha256(material).digest()))
    if not KEY_PATH.exists():
        with NamedTemporaryFile(dir=KEY_PATH.parent, prefix=".repf-llm-tmp-", delete=False) as f:
            pending = Path(f.name)
            f.write(Fernet.generate_key())
        try:
            KEY_PATH.hardlink_to(pending)
        except FileExistsError:
            pass
        finally:
            pending.unlink(missing_ok=True)
    return Fernet(KEY_PATH.read_bytes())


def encrypt(key: str, settings: Settings) -> str:
    return _cipher(settings).encrypt(key.encode()).decode()


async def active_profile(session: AsyncSession, user_id: str) -> LLMCheckpoint | LLMProfile | None:
    selection = await session.get(LLMSelection, user_id)
    if selection is None:
        return await session.get(LLMProfile, user_id)
    if selection.checkpoint_id is None:
        return None
    checkpoint = await session.get(LLMCheckpoint, selection.checkpoint_id)
    return checkpoint if checkpoint and checkpoint.user_id == user_id else None


def effective_settings(settings: Settings, profile: LLMCheckpoint | LLMProfile | None) -> Settings:
    if profile is None:
        return settings
    if profile.provider == "compatible" and profile.endpoint not in endpoints(settings).values():
        raise ValueError("Сохранённый сервер API больше недоступен. Выберите новый адрес в настройках.")
    key = None
    if profile.encrypted_key:
        try:
            key = SecretStr(_cipher(settings).decrypt(profile.encrypted_key.encode()).decode())
        except (InvalidToken, ValueError, OSError) as error:
            raise ValueError("Не удалось прочитать API-ключ. Сохраните ключ заново в настройках.") from error
    return settings.model_copy(update={
        "LLM_PROVIDER": "auto" if profile.provider == "local" else profile.provider,
        "LLM_API_KEY": key, "DEEPSEEK_API_KEY": None, "COMPATIBLE_LLM_KEY": None,
        "DEEPSEEK_MODEL": profile.model or settings.DEEPSEEK_MODEL,
        "COMPATIBLE_LLM_MODEL": profile.model or settings.COMPATIBLE_LLM_MODEL,
        "LLM_MODEL": profile.model or settings.LLM_MODEL,
        "COMPATIBLE_LLM_URL": profile.endpoint if profile.provider == "compatible" else None,
        "OLLAMA_URL": settings.OLLAMA_URL if profile.provider == "ollama" else None,
    })
