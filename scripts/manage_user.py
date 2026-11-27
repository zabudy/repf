"""Assign a role to an existing account."""
import argparse
import asyncio

from sqlalchemy import update

from backend.db.models import User
from backend.db.session import SessionLocal


async def assign(email: str, role: str) -> None:
    async with SessionLocal.begin() as session:
        changed = await session.execute(update(User).where(User.email == email.casefold()).values(role=role))
        if changed.rowcount != 1:
            raise SystemExit("Аккаунт не найден")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("email")
    parser.add_argument("role", choices=["user", "editor", "admin"])
    args = parser.parse_args()
    asyncio.run(assign(args.email, args.role))
