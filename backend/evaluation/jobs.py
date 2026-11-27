"""Durable queue for batch evaluation through the API."""
from datetime import UTC, datetime, timedelta
import logging

from sqlalchemy import and_, or_, select, update

from backend.core.config import Settings
from backend.db.models import EvaluationRun
from backend.db.session import SessionLocal
from backend.evaluation.runner import EvalParameters, evaluate

logger = logging.getLogger(__name__)


async def evaluate_once(settings: Settings) -> bool:
    now = datetime.now(UTC)
    eligible = or_(EvaluationRun.status == "pending",
                   and_(EvaluationRun.status == "running",
                        EvaluationRun.started_at < now - timedelta(hours=4)))
    async with SessionLocal.begin() as session:
        item = await session.scalar(select(EvaluationRun.id).where(eligible)
                                    .order_by(EvaluationRun.created_at).limit(1))
        if not item:
            return False
        query = await session.execute(update(EvaluationRun).where(EvaluationRun.id == item, eligible).values(
            status="running", started_at=now, generation=EvaluationRun.generation + 1,
        ).returning(EvaluationRun.parameters, EvaluationRun.generation))
        row = query.one_or_none()
        if row is None:
            return False
    try:
        result = await evaluate(EvalParameters.model_validate(row.parameters), settings)
        status = "completed"
    except Exception:
        logger.exception("Evaluation failed id=%s", item)
        result, status = {"error": "Ошибка расчёта эксперимента"}, "failed"
    async with SessionLocal.begin() as session:
        await session.execute(update(EvaluationRun).where(EvaluationRun.id == item,
                              EvaluationRun.generation == row.generation).values(status=status, result=result))
    return True
