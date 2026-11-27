"""Docker worker and periodic dispatch of persisted jobs."""
import asyncio

from celery import Celery

from backend.core.config import get_settings
from backend.db.session import engine

settings = get_settings()
celery = Celery("repf", broker=settings.REDIS_URL)
celery.conf.update(
    task_serializer="json", accept_content=["json"], task_ignore_result=True,
    broker_connection_retry_on_startup=True, worker_prefetch_multiplier=1,
    beat_schedule={
        "expire-cache": {"task": "repf.expire_cache", "schedule": 3600},
        "dispatch-forecast": {"task": "repf.forecast", "schedule": 2., "options": {"expires": 12}},
        "dispatch-evaluation": {"task": "repf.evaluate", "schedule": 8., "options": {"expires": 20}},
    },
    task_routes={"repf.evaluate": {"queue": "evaluation"}, "repf.forecast": {"queue": "forecasts"},
                 "repf.expire_cache": {"queue": "evaluation"}},
)


async def _run(awaitable) -> None:
    try:
        await awaitable
    finally:
        await engine.dispose()


@celery.task(name="repf.forecast")
def forecast() -> None:
    from backend.worker.tasks import work_once

    asyncio.run(_run(work_once(settings)))


@celery.task(name="repf.evaluate")
def evaluate() -> None:
    from backend.evaluation.jobs import evaluate_once

    asyncio.run(_run(evaluate_once(settings)))


@celery.task(name="repf.expire_cache")
def expire_cache() -> None:
    from backend.data.cache import DatabaseCache

    asyncio.run(_run(DatabaseCache().prune()))
