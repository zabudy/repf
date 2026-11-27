import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes import auth, health
from backend.core.config import get_settings
from backend.db.session import init_db
from backend.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()  # создаёт таблицы, если их нет
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=f"{settings.PROJECT_NAME} — прогноз цен на недвижимость",
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(auth.router)

    logging.getLogger(__name__).info("Приложение %s запущено", settings.PROJECT_NAME)
    return app


app = create_app()
