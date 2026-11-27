"""Create and inspect reproducible comparison jobs."""
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from backend.api.deps import CurrentUser, Session
from backend.db.models import EvaluationRun, GlossaryTerm
from backend.evaluation.runner import EvalParameters

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


@router.post("/evaluate/batch", status_code=202)
async def submit(params: EvalParameters, user: CurrentUser, session: Session) -> dict:
    try:
        params.check_values()
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if params.configured_llm and user.role != "admin":
        raise HTTPException(403, "Запуск оценки с внешней LLM доступен администратору")
    row = EvaluationRun(id=str(uuid4()), owner_id=user.id, parameters=params.model_dump(), status="pending")
    session.add(row)
    await session.commit()
    return {"evaluation_id": row.id, "status": row.status}


@router.get("/evaluate/compare")
async def compare(user: CurrentUser, session: Session, evaluation_id: UUID | None = None) -> dict:
    query = select(EvaluationRun).where(EvaluationRun.owner_id == user.id)
    if evaluation_id is not None:
        query = query.where(EvaluationRun.id == str(evaluation_id))
    row = await session.scalar(query.order_by(EvaluationRun.created_at.desc()).limit(1))
    if row is None:
        raise HTTPException(404, "Нет эксперимента")
    result = {key: value for key, value in (row.result or {}).items() if key != "rows"}
    return {"evaluation_id": row.id, "status": row.status, **result}


@router.get("/glossary")
async def glossary(session: Session, query: str = "") -> list[dict]:
    q = select(GlossaryTerm).where(GlossaryTerm.term.ilike(f"%{query[:80]}%")).order_by(GlossaryTerm.term).limit(25)
    return [{"term": row.term, "definition": row.definition} for row in (await session.scalars(q)).all()]
