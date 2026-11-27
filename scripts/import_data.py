"""Idempotent import of case and glossary artifacts into SQL and optional Qdrant."""
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.core.config import get_settings
from backend.db.models import GlossaryTerm, KnowledgeCase
from backend.db.session import SessionLocal, engine, init_db
from backend.retrieval.hybrid import CASE_PATH, CaseIndex, publish_vectors


async def import_data(path: Path, qdrant: bool = True) -> tuple[int, int]:
    cases = json.loads(path.read_text())
    glossary = json.loads(path.with_name("glossary.json").read_text())
    await init_db()
    insert = sqlite_insert if engine.dialect.name == "sqlite" else postgres_insert
    async with SessionLocal.begin() as session:
        for offset in range(0, len(cases), 100):
            items = [{"id": row["id"], "task_type": row["metadata"]["task_type"], "payload": row}
                     for row in cases[offset:offset + 100]]
            query = insert(KnowledgeCase).values(items)
            await session.execute(query.on_conflict_do_update(
                index_elements=["id"], set_={"payload": query.excluded.payload,
                                            "task_type": query.excluded.task_type}))
        for offset in range(0, len(glossary), 100):
            query = insert(GlossaryTerm).values(glossary[offset:offset + 100])
            await session.execute(query.on_conflict_do_update(index_elements=["term"],
                                                             set_={"definition": query.excluded.definition}))
    if qdrant and get_settings().QDRANT_URL:
        await publish_vectors(await asyncio.to_thread(CaseIndex.build, cases), get_settings())
    return len(cases), len(glossary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=CASE_PATH)
    parser.add_argument("--sql-only", action="store_true")
    args = parser.parse_args()
    n, m = asyncio.run(import_data(args.path, not args.sql_only))
    print(f"Imported {n} cases and {m} terms")
