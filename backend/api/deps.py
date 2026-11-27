"""Authentication and ownership dependencies."""
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.db.models import User
from backend.db.session import get_session
from backend.utils.security import token_user

Session = Annotated[AsyncSession, Depends(get_session)]
bearer = HTTPBearer(auto_error=False)


async def current_user(session: Session, auth: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> User:
    user_id = token_user(auth.credentials, get_settings()) if auth else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(401, "Войдите в систему", headers={"WWW-Authenticate": "Bearer"})
    return user


CurrentUser = Annotated[User, Depends(current_user)]
