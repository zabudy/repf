import logging

from fastapi import FastAPI

from backend.api.routes import health
from backend.core.config import get_settings
from backend.utils.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=f"{settings.PROJECT_NAME} — прогноз цен на недвижимость",
        version=settings.VERSION,
        debug=settings.DEBUG,
    )
    app.include_router(health.router)

    logging.getLogger(__name__).info("Приложение %s запущено", settings.PROJECT_NAME)
    return app


app = create_app()
