"""ASGI application. Local mode runs the durable queue beside the API."""
import asyncio
from contextlib import asynccontextmanager, suppress
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import auth, evaluate, forecast, health, llm
from backend.core.config import get_settings
from backend.data.cache import DatabaseCache
from backend.db.session import engine, init_db
from backend.utils.logging import setup_logging
from backend.worker.tasks import work_forever

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.WORKER_MODE == "celery" and (
        settings.JWT_SECRET_KEY is None or len(settings.JWT_SECRET_KEY.get_secret_value()) < 32
    ):
        raise RuntimeError("JWT_SECRET_KEY must have at least 32 characters in Celery mode")
    await init_db()
    await auth.ensure_admin(settings)
    await DatabaseCache().prune()
    worker = asyncio.create_task(work_forever(settings)) if settings.WORKER_MODE == "local" else None
    try:
        yield
    finally:
        if worker:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    app = FastAPI(title=f"{settings.PROJECT_NAME} — прогноз цен на недвижимость",
                  version=settings.VERSION, debug=settings.DEBUG, lifespan=lifespan)
    for router in (health.router, auth.router, forecast.router, evaluate.router, llm.router):
        app.include_router(router)
    app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")

    @app.exception_handler(RequestValidationError)
    async def redact_llm_validation(request: Request, error: RequestValidationError):
        if not request.url.path.startswith("/api/v1/llm/"):
            return await request_validation_exception_handler(request, error)
        fields = ("loc", "msg", "type")
        return JSONResponse(status_code=422, content={
            "detail": [{key: item[key] for key in fields} for item in error.errors()],
        })

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(FRONTEND / "index.html")

    logging.getLogger(__name__).info("Приложение %s настроено", settings.PROJECT_NAME)
    return app


app = create_app()
