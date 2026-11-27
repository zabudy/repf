"""Настройки приложения: читаются из переменных окружения и файла .env."""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # в .env есть переменные для будущих этапов — игнорируем
    )

    DEBUG: bool = True
    PROJECT_NAME: str = "REPF"
    VERSION: str = "0.1.0"
    DEFAULT_REGION: str = "Kazan"
    FORECAST_HORIZON_MONTHS: int = 12
    CONFIDENCE_THRESHOLD: float = 0.6
    HITL_ENABLED: bool = True

    DATABASE_URL: str = "sqlite+aiosqlite:///./repf.db"

    LLM_PROVIDER: Literal["auto", "deepseek", "ollama", "compatible", "mock"] = "auto"
    LLM_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_MODEL: str = "deepseek-flash"
    OLLAMA_URL: str | None = None
    LLM_MODEL: str = "qwen2.5:7b"
    COMPATIBLE_LLM_URL: str | None = None
    COMPATIBLE_LLM_KEY: SecretStr | None = None
    COMPATIBLE_LLM_MODEL: str = "qwen2.5"
    LLM_TIMEOUT: float = Field(default=18, gt=0, le=120)
    STATISTICAL_DIAGNOSTICS: bool = True
    MAX_ITERATIONS: int = Field(default=3, ge=1, le=3)

    WORKER_MODE: Literal["local", "celery", "disabled"] = "local"
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    QDRANT_URL: str | None = None
    QDRANT_COLLECTION: str = "repf_cases_v1"

    JWT_SECRET_KEY: SecretStr | None = None
    @field_validator("JWT_SECRET_KEY", mode="before")
    @classmethod
    def empty_secret(cls, value):
        return value or None

    JWT_EXPIRATION_MINUTES: int = Field(default=480, ge=5, le=10080)
    ADMIN_EMAIL: str | None = None
    ADMIN_PASSWORD: SecretStr | None = None

    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
