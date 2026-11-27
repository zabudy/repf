"""Durable forecast queue and trace persistence shared by local and Celery workers."""
import asyncio
from datetime import UTC, datetime, timedelta
import logging
from uuid import uuid4

from sqlalchemy import and_, or_, select, update

from backend.agents.base import Trace
from backend.agents.supervisor import Supervisor
from backend.core.config import Settings
from backend.core.forecast import ForecastRequest
from backend.core.llm.manager import LLMManager
from backend.core.llm.profile import active_profile, effective_settings
from backend.data.cache import DatabaseCache, cache_key
from backend.db.models import ForecastTask, LLMCall, ReasoningLog
from backend.db.session import SessionLocal
from backend.retrieval.hybrid import HybridRetriever, load_cases

logger = logging.getLogger(__name__)
LEASE = timedelta(minutes=25)


class DatabaseTrace:
    def __init__(self, task_id: str):
        self.task_id = task_id

    async def span(self, entry: dict) -> None:
        async with SessionLocal.begin() as session:
            if entry["status"] == "running":
                session.add(ReasoningLog(task_id=self.task_id, **entry))
            else:
                await session.execute(update(ReasoningLog).where(ReasoningLog.id == entry["id"]).values(
                    status=entry["status"], details=entry["details"],
                ))

    async def llm_call(self, span_id: str, entry: dict) -> None:
        async with SessionLocal.begin() as session:
            session.add(LLMCall(id=str(uuid4()), task_id=self.task_id, span_id=span_id, details=entry))


async def claim() -> tuple[str, int, dict, dict | None, str] | None:
    now = datetime.now(UTC)
    allowed = or_(ForecastTask.status == "pending",
                  and_(ForecastTask.status == "running", ForecastTask.started_at < now - LEASE))
    async with SessionLocal.begin() as session:
        task_id = await session.scalar(select(ForecastTask.id).where(allowed)
                                       .order_by(ForecastTask.created_at).limit(1))
        if task_id is None:
            return None
        result = await session.execute(update(ForecastTask).where(ForecastTask.id == task_id, allowed).values(
            status="running", started_at=now, generation=ForecastTask.generation + 1,
        ).returning(ForecastTask.id, ForecastTask.generation, ForecastTask.request, ForecastTask.review, ForecastTask.owner_id))
        row = result.one_or_none()
        return tuple(row) if row is not None else None


async def finish(task_id: str, generation: int, **values) -> None:
    async with SessionLocal.begin() as session:
        await session.execute(update(ForecastTask).where(
            ForecastTask.id == task_id, ForecastTask.generation == generation, ForecastTask.status == "running",
        ).values(**values))


async def work_once(settings: Settings) -> bool:
    job = await claim()
    if job is None:
        return False
    task_id, generation, payload, review, owner_id = job
    try:
        request = ForecastRequest.model_validate(payload)
        index = await asyncio.to_thread(load_cases) if request.mode != "baseline" else None
        cache, sink = DatabaseCache(), DatabaseTrace(task_id)
        key = cache_key("forecast", {"owner": owner_id, "request": payload,
                                     "case_version": index.fingerprint if index else None,
                                     "version": 1, "llm": settings.LLM_PROVIDER,
                                     "threshold": settings.CONFIDENCE_THRESHOLD,
                                     "statistical": settings.STATISTICAL_DIAGNOSTICS})
        # The deterministic baseline can safely reuse a result. Full agent runs
        # produce a fresh trace and may reuse only schema-validated LLM replies.
        stored = await cache.get(key) if request.mode == "baseline" and review is None else None
        if stored is not None:
            async with Trace(sink).step("ForecastCache") as (_, summary):
                summary["hit"] = True
            result = stored["result"]
            result["trajectory"] = {"iterations": 0, "llm_calls": 0, "tokens": 0,
                                    "estimated_calls": 0, "latency_ms": 0, "cache_hit": True}
            await finish(task_id, generation, status="completed", result=result,
                         state={**stored["state"], "task_id": task_id}, error=None)
            return True
        llm_settings = settings
        cache_scope = ""
        if request.mode == "+agents":
            async with SessionLocal() as session:
                profile = await active_profile(session, owner_id)
            llm_settings = effective_settings(settings, profile)
            if profile is not None:
                cache_scope = owner_id + (profile.encrypted_key or "")
        manager = LLMManager(llm_settings, cache=cache, cache_scope=cache_scope)
        runner = Supervisor(settings, HybridRetriever(index, settings), manager)
        outcome = await runner.run(task_id, request, sink, review)
        status = "waiting_for_human" if outcome.needs_review else "completed"
        snapshot = outcome.state.model_dump(mode="json", exclude={"prepared_id", "result_id", "input_id"})
        await finish(task_id, generation, result=outcome.result, state=snapshot,
                     status=status, error=None)
        if request.mode == "baseline" and review is None:
            await cache.put(key, {"state": snapshot, "result": outcome.result})
    except asyncio.CancelledError:
        await finish(task_id, generation, status="pending")
        raise
    except (ValueError, ArithmeticError) as error:
        message = str(error)[:400]
        await finish(task_id, generation, status="failed", error=message)
        logger.warning("Forecast rejected task=%s type=%s", task_id, type(error).__name__)
    except Exception:
        logger.exception("Forecast failed task=%s", task_id)
        await finish(task_id, generation, status="failed", error="Внутренняя ошибка расчёта")
    return True


async def work_forever(settings: Settings) -> None:
    from backend.evaluation.jobs import evaluate_once

    while True:
        try:
            if await work_once(settings) or await evaluate_once(settings):
                continue
        except Exception:
            logger.exception("Queue iteration failed")
        await asyncio.sleep(.5)
