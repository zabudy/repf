"""User-owned model checkpoints and live availability verification."""
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
from pydantic import Field, field_validator
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from backend.api.deps import CurrentUser, Session
from backend.core.config import get_settings
from backend.core.forecast import InputModel, ToolChoice
from backend.core.llm.manager import LLMManager, providers
from backend.core.llm.profile import ProfileInput, active_profile, effective_settings, encrypt, endpoints
from backend.core.llm.tools import validate
from backend.db.models import LLMCheckpoint, LLMProfile, LLMSelection
from backend.db.session import engine

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


class CheckpointInput(ProfileInput):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class SelectionInput(InputModel):
    checkpoint_id: str | None


def _public(profile: LLMCheckpoint | LLMProfile) -> dict:
    return {"provider": profile.provider, "model": profile.model, "endpoint": profile.endpoint,
            "has_api_key": bool(profile.encrypted_key)}


async def describe(session: Session, user_id: str) -> dict:
    settings = get_settings()
    defaults = providers(settings)
    provider = defaults[0] if defaults else None
    default = {"provider": provider.name, "model": provider.model} if provider else None
    if provider and provider.name == "compatible":
        default["endpoint"] = settings.COMPATIBLE_LLM_URL.rstrip("/")
    active = await active_profile(session, user_id)
    checkpoints = await session.scalars(select(LLMCheckpoint).where(LLMCheckpoint.user_id == user_id)
                                        .order_by(LLMCheckpoint.name, LLMCheckpoint.id))
    entries = [{"id": row.id, "name": row.name, **_public(row)} for row in checkpoints]
    legacy = await session.get(LLMProfile, user_id)
    if legacy:
        entries.insert(0, {"id": "legacy", "name": "Ранее сохранённая модель", **_public(legacy)})
    active_id = active.id if isinstance(active, LLMCheckpoint) else "legacy" if active else None
    return {
        "profile": _public(active) if active else None,
        "checkpoints": entries,
        "active_checkpoint_id": active_id,
        "default": default,
        "endpoints": [{"label": label, "url": url} for label, url in endpoints(settings).items()],
        "models": {"deepseek": settings.DEEPSEEK_MODEL, "ollama": settings.LLM_MODEL,
                   "compatible": settings.COMPATIBLE_LLM_MODEL},
    }


def _configuration(payload: ProfileInput, prior: LLMProfile | None = None) -> dict:
    settings = get_settings()
    model = ((payload.model or "").strip() or None) if payload.provider != "local" else None
    endpoint = payload.endpoint if payload.provider == "compatible" else None
    if payload.provider == "compatible" and endpoint not in endpoints(settings).values():
        raise HTTPException(422, "Выберите адрес из списка API.")
    requires_model = payload.provider == "compatible" or (
        isinstance(payload, CheckpointInput) and payload.provider != "local"
    )
    if requires_model and model is None:
        raise HTTPException(422, "Укажите точный идентификатор модели у провайдера.")

    key = None
    if prior and (prior.provider, prior.endpoint) == (payload.provider, endpoint):
        key = prior.encrypted_key
    if payload.provider in {"deepseek", "compatible"} and payload.api_key is not None:
        raw = payload.api_key.get_secret_value().strip()
        if len(raw) > 2048 or any(char.isspace() for char in raw):
            raise HTTPException(422, "Проверьте формат API-ключа.")
        key = encrypt(raw, settings) if raw else None
    if payload.provider == "deepseek" and not key:
        raise HTTPException(422, "Введите API-ключ DeepSeek.")
    return {"provider": payload.provider, "model": model, "endpoint": endpoint, "encrypted_key": key}


def _insert(model):
    return sqlite_insert(model) if engine.dialect.name == "sqlite" else postgres_insert(model)


async def _choose(session: Session, user_id: str, checkpoint_id: str | None) -> None:
    if checkpoint_id == "legacy":
        await session.execute(delete(LLMSelection).where(LLMSelection.user_id == user_id))
    else:
        query = _insert(LLMSelection).values(user_id=user_id, checkpoint_id=checkpoint_id)
        await session.execute(query.on_conflict_do_update(index_elements=["user_id"],
                                                           set_={"checkpoint_id": checkpoint_id}))


async def _owned(session: Session, user_id: str, checkpoint_id: str) -> LLMCheckpoint | LLMProfile:
    model, pk = (LLMProfile, user_id) if checkpoint_id == "legacy" else (LLMCheckpoint, checkpoint_id)
    row = await session.get(model, pk)
    if row is None or row.user_id != user_id:
        raise HTTPException(404, "Пресет не найден.")
    return row


async def _probe(profile: LLMCheckpoint | LLMProfile | None) -> dict:
    try:
        settings = effective_settings(get_settings(), profile)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    manager = LLMManager(settings)
    if not manager.providers:
        raise HTTPException(422, "Выберите провайдера для проверки подключения.")
    manager.providers = manager.providers[:1]
    calls = []

    async def record(entry: dict) -> None:
        calls.append(entry)

    fallback = validate(ToolChoice(name="fit_holt"), "timeseries")
    _, source = await manager.select("timeseries", ["fit_holt"], {"horizon": 2}, fallback, record,
                                     native_only=True)
    call = calls[0]
    if source == "fallback":
        error = call.get("error")
        if call["status"] == "invalid_output":
            message = "Модель ответила, но не вернула корректный вызов инструмента."
        elif error in {"HTTP 401", "HTTP 403"}:
            message = "Провайдер отклонил ключ или доступ к модели."
        elif error == "HTTP 404":
            message = "Модель не найдена или недоступна для этого ключа."
        elif error in {"HTTP 400", "HTTP 422"}:
            message = "Провайдер отклонил запрос. Проверьте идентификатор модели и поддержку Tool Calling."
        elif error == "HTTP 429":
            message = "Провайдер вернул ограничение по запросам или балансу."
        else:
            message = "Провайдер недоступен или не принял запрос."
        raise HTTPException(502, message)
    return {field: call[field] for field in ("provider", "model", "input_tokens", "output_tokens", "latency_ms")}


@router.get("/settings")
async def read(user: CurrentUser, session: Session) -> dict:
    return await describe(session, user.id)


@router.put("/settings")
async def save(payload: ProfileInput, user: CurrentUser, session: Session) -> dict:
    """Keep the original API for integrations using one profile."""
    user_id = user.id
    values = _configuration(payload, await session.get(LLMProfile, user_id))
    await session.rollback()
    if payload.provider != "local":
        await _probe(LLMProfile(user_id=user_id, **values))
    query = _insert(LLMProfile).values(user_id=user_id, **values)
    await session.execute(query.on_conflict_do_update(index_elements=["user_id"], set_=values))
    await _choose(session, user_id, "legacy")
    await session.commit()
    return await describe(session, user_id)


@router.delete("/settings", status_code=204)
async def reset(user: CurrentUser, session: Session) -> Response:
    user_id = user.id
    await session.execute(delete(LLMProfile).where(LLMProfile.user_id == user_id))
    await _choose(session, user_id, None)
    await session.commit()
    return Response(status_code=204)


@router.post("/probe")
async def probe(user: CurrentUser, session: Session) -> dict:
    profile = await active_profile(session, user.id)
    if profile is not None:
        session.expunge(profile)
    await session.rollback()
    return await _probe(profile)


@router.post("/checkpoints", status_code=201)
async def create_checkpoint(payload: CheckpointInput, user: CurrentUser, session: Session) -> dict:
    user_id = user.id
    values = _configuration(payload)
    if await session.scalar(select(LLMCheckpoint.id).where(
        LLMCheckpoint.user_id == user_id, LLMCheckpoint.name == payload.name,
    )):
        raise HTTPException(409, "Пресет с таким названием уже есть.")
    await session.rollback()
    checkpoint = LLMCheckpoint(id=str(uuid4()), user_id=user_id, name=payload.name, **values)
    if checkpoint.provider != "local":
        await _probe(checkpoint)
    session.add(checkpoint)
    try:
        await session.commit()
    except IntegrityError as error:
        raise HTTPException(409, "Не удалось сохранить пресет с таким названием.") from error
    return {"id": checkpoint.id, "name": checkpoint.name, **_public(checkpoint)}


@router.put("/selection")
async def select_checkpoint(payload: SelectionInput, user: CurrentUser, session: Session) -> dict:
    user_id, checkpoint_id = user.id, payload.checkpoint_id
    if checkpoint_id is not None:
        profile = await _owned(session, user_id, checkpoint_id)
        if profile.provider != "local":
            session.expunge(profile)
            await session.rollback()
            await _probe(profile)
            await _owned(session, user_id, checkpoint_id)
    await _choose(session, user_id, checkpoint_id)
    try:
        await session.commit()
    except IntegrityError as error:
        raise HTTPException(404, "Пресет больше недоступен.") from error
    return await describe(session, user_id)


@router.post("/checkpoints/{checkpoint_id}/probe")
async def probe_checkpoint(checkpoint_id: str, user: CurrentUser, session: Session) -> dict:
    profile = await _owned(session, user.id, checkpoint_id)
    session.expunge(profile)
    await session.rollback()
    return await _probe(profile)


@router.delete("/checkpoints/{checkpoint_id}", status_code=204)
async def delete_checkpoint(checkpoint_id: str, user: CurrentUser, session: Session) -> Response:
    user_id = user.id
    checkpoint = await _owned(session, user_id, checkpoint_id)
    if checkpoint_id != "legacy":
        await session.execute(update(LLMSelection).where(
            LLMSelection.user_id == user_id, LLMSelection.checkpoint_id == checkpoint_id,
        ).values(checkpoint_id=None))
    await session.delete(checkpoint)
    await session.commit()
    return Response(status_code=204)
