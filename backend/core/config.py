"""Настройки приложения: читаются из переменных окружения и файла .env."""
from functools import lru_cache

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

    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
