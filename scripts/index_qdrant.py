"""Publish the current case embeddings to the configured Qdrant collection."""
import asyncio

from backend.core.config import get_settings
from backend.retrieval.hybrid import load_cases, publish_vectors


async def main() -> None:
    index = await asyncio.to_thread(load_cases)
    if index is None:
        raise RuntimeError("Generate the case base first")
    await publish_vectors(index, get_settings())
    print(f"Indexed {len(index.cases)} cases")


if __name__ == "__main__":
    asyncio.run(main())
