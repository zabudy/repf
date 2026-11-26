from fastapi import APIRouter

from backend.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    """Проверка, что сервис жив."""
    s = get_settings()
    return {"status": "ok", "project": s.PROJECT_NAME, "version": s.VERSION}
