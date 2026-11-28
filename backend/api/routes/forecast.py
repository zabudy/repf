"""Queue forecasts, expose the trace and apply a recorded HITL decision."""
import csv
import io
import logging
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import Field, StringConstraints, ValidationError
from sqlalchemy import select, update
from starlette.datastructures import UploadFile

from backend.api.deps import CurrentUser, Session
from backend.core.config import get_settings
from backend.core.forecast import ForecastRequest, InputModel, Review, ToolChoice
from backend.core.llm.manager import LLMManager
from backend.core.llm.profile import active_profile, effective_settings
from backend.core.llm.tools import available, validate
from backend.db.models import ForecastTask, KnowledgeCase, LLMCall, ReasoningLog

router = APIRouter(prefix="/api/v1/forecast", tags=["forecasts"])
LIMIT = 5 * 1024 * 1024
logger = logging.getLogger(__name__)

ChatQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)]


class ChatTurn(InputModel):
    question: ChatQuestion
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6000)]


class ForecastChat(InputModel):
    question: ChatQuestion
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)


async def get_task(task_id: UUID, user, session, can_review: bool = False) -> ForecastTask:
    task = await session.get(ForecastTask, str(task_id))
    if task is None or (task.owner_id != user.id and user.role != "admin" and
                        not (can_review and user.role == "editor")):
        raise HTTPException(404, "Задача не найдена")
    return task


def _csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames or not {"date", "value"}.issubset(reader.fieldnames):
        raise ValueError("CSV должен содержать колонки date и value")
    observations = []
    for row in reader:
        if len(observations) >= 10000:
            raise ValueError("В CSV допускается до 10 000 строк")
        observations.append({"date": row["date"], "value": row["value"] or None})
    return observations


async def _payload(request: Request) -> ForecastRequest:
    try:
        if request.headers.get("content-type", "").startswith("multipart/"):
            async with request.form(max_files=1, max_fields=6) as form:
                file = form.get("file")
                if not isinstance(file, UploadFile):
                    raise ValueError("Добавьте CSV в поле file")
                content = await file.read(LIMIT + 1)
                if len(content) > LIMIT:
                    raise HTTPException(413, "Файл больше 5 МБ")
                data = {"task_type": "timeseries", "observations": _csv(content),
                        "frequency": form.get("frequency") or None,
                        "nonnegative": form.get("nonnegative", "false")}
                if form.get("seasonal_period"):
                    data["seasonal_period"] = form["seasonal_period"]
                return ForecastRequest.model_validate({"data": data, "horizon": form.get("horizon", 6),
                                                       "mode": form.get("mode", "+agents")})
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > LIMIT:
                raise HTTPException(413, "Запрос больше 5 МБ")
        return ForecastRequest.model_validate_json(bytes(chunks))
    except ValidationError as error:
        raise HTTPException(422, error.errors(include_context=False, include_url=False, include_input=False)) from error
    except (ValueError, csv.Error) as error:
        raise HTTPException(422, str(error)[:300]) from error


@router.post("/", status_code=202, openapi_extra={"requestBody": {"required": True, "content": {
    "application/json": {"schema": {"type": "object", "required": ["data"],
                                    "properties": {"data": {"type": "object"}, "horizon": {"type": "integer"},
                                                   "mode": {"enum": ["baseline", "+rag", "+agents"]}}}},
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"],
                                       "properties": {"file": {"type": "string", "format": "binary"},
                                                      "horizon": {"type": "integer"},
                                                      "frequency": {"type": "string"}}}},
}}})
async def submit(request: Request, user: CurrentUser, session: Session) -> dict:
    payload = await _payload(request)
    task = ForecastTask(id=str(uuid4()), owner_id=user.id, status="pending", request=payload.model_dump(mode="json"))
    session.add(task)
    await session.commit()
    return {"task_id": task.id, "status": task.status}


@router.get("/")
async def list_tasks(user: CurrentUser, session: Session, review_queue: bool = False) -> list[dict]:
    query = select(ForecastTask).order_by(ForecastTask.created_at.desc()).limit(50)
    if not review_queue or user.role not in {"admin", "editor"}:
        query = query.where(ForecastTask.owner_id == user.id)
    if review_queue:
        query = query.where(ForecastTask.status == "waiting_for_human")
    return [{"task_id": row.id, "status": row.status, "task_type": row.request["data"]["task_type"],
             "horizon": row.request["horizon"], "created_at": row.created_at.isoformat()}
            for row in (await session.scalars(query)).all()]


@router.get("/{task_id}/status")
async def status(task_id: UUID, user: CurrentUser, session: Session) -> dict:
    task = await get_task(task_id, user, session, can_review=True)
    return {"task_id": task.id, "status": task.status, "error": task.error, "review": task.review,
            "horizon": task.request["horizon"], "models": available(task.request["data"]["task_type"])}


@router.get("/{task_id}/result")
async def result(task_id: UUID, user: CurrentUser, session: Session) -> dict:
    task = await get_task(task_id, user, session, can_review=True)
    if task.result is None or task.status not in {"completed", "waiting_for_human"}:
        raise HTTPException(409, "Результат ещё недоступен")
    return {**task.result, "task_id": task.id, "status": task.status, "review": task.review,
            "chat_enabled": task.owner_id == user.id and task.request.get("mode") == "+agents",
            "download_url": f"/api/v1/forecast/{task.id}/download"}


@router.post("/{task_id}/chat")
async def chat(task_id: UUID, payload: ForecastChat, user: CurrentUser, session: Session) -> dict:
    task = await get_task(task_id, user, session)
    if task.owner_id != user.id:
        raise HTTPException(404, "Задача не найдена")
    if task.request.get("mode") != "+agents":
        raise HTTPException(409, "Чат доступен для прогноза в режиме «Агенты + RAG»")
    if task.result is None or task.status not in {"completed", "waiting_for_human"}:
        raise HTTPException(409, "Результат ещё недоступен")
    context = {key: task.result[key] for key in (
        "model", "frequency", "dates", "predictions", "interval", "metrics", "validation",
        "data_summary", "data_origin", "confidence", "explanation", "needs_review",
        "predicted_price", "price_per_m2", "factors",
    ) if key in task.result}
    context.update(task_type=task.request["data"]["task_type"], horizon=task.request["horizon"])
    profile = await active_profile(session, user.id)
    try:
        settings = effective_settings(get_settings(), profile)
    except ValueError as error:
        raise HTTPException(503, str(error)) from error
    await session.rollback()
    manager = LLMManager(settings)
    if not manager.providers:
        raise HTTPException(503, "Выберите доступную LLM в настройках, чтобы открыть чат")
    model = manager.providers[0].model

    async def log(entry: dict) -> None:
        logger.info("ForecastChat task=%s provider=%s model=%s status=%s input_tokens=%s output_tokens=%s latency_ms=%s",
                    task_id, entry["provider"], entry["model"], entry["status"],
                    entry["input_tokens"], entry["output_tokens"], entry["latency_ms"])

    try:
        answer = await manager.discuss_forecast(context, [turn.model_dump() for turn in payload.history],
                                                payload.question, log)
    except ValueError as error:
        raise HTTPException(502, "Модель не вернула ответ. Повторите отправку или выберите другую LLM.") from error
    return {"answer": answer, "model": model}


@router.get("/{task_id}/download")
async def download(task_id: UUID, user: CurrentUser, session: Session):
    forecast = await result(task_id, user, session)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["date", "prediction", "lower_80", "upper_80"])
    writer.writerows(zip(forecast["dates"], forecast["predictions"],
                         forecast["interval"]["lower"], forecast["interval"]["upper"], strict=True))
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="forecast-{task_id}.csv"'})


@router.get("/{task_id}/reasoning")
async def reasoning(task_id: UUID, user: CurrentUser, session: Session) -> dict:
    task = await get_task(task_id, user, session, can_review=True)
    spans = (await session.scalars(select(ReasoningLog).where(ReasoningLog.task_id == task.id)
                                   .order_by(ReasoningLog.created_at))).all()
    calls = (await session.scalars(select(LLMCall).where(LLMCall.task_id == task.id)
                                    .order_by(LLMCall.created_at))).all()
    nodes = {span.id: {"id": span.id, "parent_id": span.parent_id, "agent": span.agent,
                       "status": span.status, "details": span.details, "created_at": span.created_at.isoformat(),
                       "children": [], "llm_calls": []} for span in spans}
    roots = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"])
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
    for call in calls:
        if call.span_id in nodes:
            nodes[call.span_id]["llm_calls"].append({"id": call.id, **call.details})
    return {"task_id": task.id, "spans": roots}


@router.post("/{task_id}/hitl")
async def review(task_id: UUID, payload: Review, user: CurrentUser, session: Session) -> dict:
    task = await get_task(task_id, user, session, can_review=True)
    if task.status != "waiting_for_human":
        raise HTTPException(409, "Задача не ожидает решения")
    decision = {**payload.model_dump(exclude_none=True), "user_id": user.id}
    changes = {"review": decision}
    if payload.action == "retry":
        updated = dict(task.request)
        if payload.horizon is not None:
            updated["horizon"] = payload.horizon
        try:
            request = ForecastRequest.model_validate(updated)
            if payload.model:
                validate(ToolChoice(name=payload.model), request.data.task_type)
        except (ValidationError, ValueError) as error:
            raise HTTPException(422, "Горизонт или модель недопустимы для задачи") from error
        changes.update(status="pending", request=request.model_dump(mode="json"), result=None, error=None)
    elif payload.action == "accept":
        decision["knowledge_case_id"] = f"review:{task.id}"
        changes.update(status="completed", result={**task.result, "needs_review": False, "accepted_by_human": True})
    else:
        changes.update(status="failed", error="Результат отклонён при проверке")
    transition = await session.execute(update(ForecastTask).where(
        ForecastTask.id == task.id, ForecastTask.status == "waiting_for_human",
    ).values(**changes))
    if transition.rowcount != 1:
        await session.rollback()
        raise HTTPException(409, "Решение уже записано")
    if payload.action == "accept":
        model = task.result["model"]
        session.add(KnowledgeCase(
            id=decision["knowledge_case_id"], task_type=task.request["data"]["task_type"],
            payload={"source": "human_review", "owner_id": task.owner_id, "eligible_for_retrieval": False,
                     "model": model, "metrics": task.result["metrics"], "review": decision},
        ))
    session.add(ReasoningLog(id=str(uuid4()), task_id=task.id, agent="HumanReview", status="ok",
                             details={"output": decision}))
    await session.commit()
    return {"task_id": str(task_id), "status": changes["status"], "review": decision}


@router.get("/demo/series")
async def demo_series() -> dict:
    path = Path("data/import/tatarstan_secondary.csv")
    return {"task_type": "timeseries", "observations": _csv(path.read_bytes()),
            "frequency": "QS", "nonnegative": True}
