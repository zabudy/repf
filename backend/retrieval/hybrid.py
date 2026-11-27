"""Contextual BM25, statistical PCA embeddings, filtered dense search and RRF."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

import httpx
import numpy as np

from backend.core.config import Settings
from backend.core.forecast import TaskType

CASE_PATH = Path("data/processed/cases.json")
DIMENSIONS = 128


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


def describe(meta: dict) -> str:
    if meta.get("task_type") == "tabular":
        return (f"tabular regression price_rub {meta.get('domain', 'housing')} "
                f"{meta.get('district', '')} feature_count_{meta.get('num_features', 7)} "
                f"{'missing' if meta.get('missing_rate', 0) > .04 else 'mostly_complete'}")
    trend = meta.get("trend", 0)
    direction = "rising" if trend > .8 else "falling" if trend < -.8 else "flat"
    season = "seasonal" if meta.get("seasonality", 0) > .35 else "nonseasonal"
    persistence = "persistent" if meta.get("autocorrelation", 0) > .5 else "weakly_correlated"
    return f"timeseries {meta.get('frequency', '')} {direction} {season} {persistence}"


@dataclass(frozen=True)
class CaseIndex:
    cases: list[dict]
    matrices: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    words: list[Counter]
    fingerprint: str

    @classmethod
    def build(cls, cases: list[dict]) -> "CaseIndex":
        centers, scales, rotations, matrices = {}, {}, {}, {}
        for task_type in {row["metadata"]["task_type"] for row in cases}:
            x = np.asarray([row["features"] for row in cases if row["metadata"]["task_type"] == task_type])
            center = x.mean(axis=0)
            scale = np.maximum(x.std(axis=0), 1e-8)
            z = (x - center) / scale
            # Right singular vectors of z are eigenvectors of its feature covariance.
            # The smaller matrix avoids a full decomposition over all stored cases.
            _, directions = np.linalg.eigh(z.T @ z)
            dimensions = min(DIMENSIONS, min(z.shape))
            rotation = directions[:, -dimensions:][:, ::-1]
            result = (z @ rotation)
            if result.shape[1] < DIMENSIONS:
                result = np.pad(result, ((0, 0), (0, DIMENSIONS - result.shape[1])))
            matrices[task_type] = result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
            centers[task_type], scales[task_type], rotations[task_type] = center, scale, rotation
        words = [Counter(tokenize(f"{describe(row['metadata'])} {row['description']}")) for row in cases]
        fingerprint = sha256(b"covariance-pca-v1:" + json.dumps(cases, sort_keys=True).encode()).hexdigest()[:16]
        return cls(cases, matrices, centers, scales, rotations, words, fingerprint)

    def vector(self, task_type: TaskType, features: np.ndarray) -> np.ndarray:
        transformed = ((features - self.centers[task_type]) / self.scales[task_type]) @ self.rotations[task_type]
        transformed = np.pad(transformed, (0, DIMENSIONS - len(transformed)))
        return transformed / max(np.linalg.norm(transformed), 1e-12)

    def matches(self, meta: dict, exclude_id: str | None) -> list[int]:
        fields = ("task_type", "frequency") if meta["task_type"] == "timeseries" else (
            "task_type", "target_column", "problem_type",
        )
        return [i for i, row in enumerate(self.cases) if row["id"] != exclude_id and
                all(row["metadata"].get(key) == meta.get(key) for key in fields)]

    def bm25(self, meta: dict, candidates: list[int]) -> list[int]:
        query = set(tokenize(describe(meta)))
        counts = [self.words[i] for i in candidates]
        avg_len = sum(sum(counter.values()) for counter in counts) / len(counts)
        idf = {word: np.log1p((len(counts) - sum(word in row for row in counts) + .5) /
                            (sum(word in row for row in counts) + .5)) for word in query}
        scores = [
            sum(idf[word] * 2.2 * doc[word] /
                (doc[word] + 1.2 * (.25 + .75 * sum(doc.values()) / avg_len))
                for word in query if doc[word])
            for doc in counts
        ]
        return [candidates[j] for j in np.argsort(-np.asarray(scores), kind="stable")[:30]]

    def dense(self, task_type: TaskType, candidates: list[int], vector: np.ndarray) -> list[int]:
        positions = {i: j for j, i in enumerate(
            k for k, row in enumerate(self.cases) if row["metadata"]["task_type"] == task_type
        )}
        matrix = self.matrices[task_type][[positions[i] for i in candidates]]
        indices = np.argsort(-(matrix @ vector), kind="stable")[:30]
        return [candidates[j] for j in indices]

    def qdrant_points(self) -> list[dict]:
        positions = Counter()
        points = []
        for row in self.cases:
            task_type = row["metadata"]["task_type"]
            points.append({"id": str(uuid5(NAMESPACE_URL, self.fingerprint + row["id"])),
                           "vector": self.matrices[task_type][positions[task_type]].tolist(),
                           "payload": {**row["metadata"], "case_id": row["id"], "version": self.fingerprint}})
            positions[task_type] += 1
        return points


@lru_cache(maxsize=2)
def _load(path: Path, stamp: int) -> CaseIndex:
    return CaseIndex.build(json.loads(path.read_text()))


def load_cases(path: Path = CASE_PATH) -> CaseIndex | None:
    return _load(path, path.stat().st_mtime_ns) if path.exists() else None


def reciprocal_rank(lists: list[list[int]], limit: int = 5) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for position, candidate in enumerate(ranked, 1):
            scores[candidate] = scores.get(candidate, 0) + 1 / (60 + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


class HybridRetriever:
    def __init__(self, index: CaseIndex | None, settings: Settings):
        self.index = index
        self.qdrant = settings.QDRANT_URL.rstrip("/") if settings.QDRANT_URL else None
        self.collection = settings.QDRANT_COLLECTION

    async def retrieve(self, metadata: dict, features: np.ndarray,
                       exclude_id: str | None = None) -> tuple[list[dict], str]:
        index = self.index
        if index is None or metadata["task_type"] not in index.matrices:
            return [], "empty"
        eligible = index.matches(metadata, exclude_id)
        if not eligible:
            return [], "empty"
        vector = index.vector(metadata["task_type"], features)
        lexical = index.bm25(metadata, eligible)
        dense, backend = None, "local_pca"
        if self.qdrant:
            try:
                dense = await self._remote(index, metadata, vector, eligible)
            except (httpx.HTTPError, KeyError, ValueError):
                dense = None
            if dense:
                backend = "qdrant"
        if not dense:
            dense = index.dense(metadata["task_type"], eligible, vector)
        output = []
        for pos, score in reciprocal_rank([lexical, dense]):
            row = index.cases[pos]
            output.append({"id": row["id"], "model": row["best_model"], "arguments": row["arguments"],
                           "metrics": row["metrics"], "context": describe(row["metadata"]),
                           "description": row["description"], "rrf_score": score})
        return output, backend

    async def _remote(self, index: CaseIndex, metadata: dict, vector: np.ndarray,
                      eligible: list[int]) -> list[int]:
        lookup = {index.cases[pos]["id"]: pos for pos in eligible}
        match = [{"key": key, "match": {"value": value}} for key, value in {
            "version": index.fingerprint, "task_type": metadata["task_type"],
            **({"frequency": metadata["frequency"]} if metadata["task_type"] == "timeseries" else
               {"target_column": metadata["target_column"], "problem_type": metadata["problem_type"]}),
        }.items()]
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.post(f"{self.qdrant}/collections/{self.collection}/points/query",
                                         json={"query": vector.tolist(), "filter": {"must": match},
                                               "limit": 30, "with_payload": True})
            response.raise_for_status()
        return [lookup[point["payload"]["case_id"]] for point in response.json()["result"]["points"]
                if point["payload"]["case_id"] in lookup]


async def publish_vectors(index: CaseIndex, settings: Settings) -> None:
    if not settings.QDRANT_URL:
        return
    url = f"{settings.QDRANT_URL.rstrip('/')}/collections/{settings.QDRANT_COLLECTION}"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        if response.status_code == 404:
            response = await client.put(url, json={"vectors": {"size": DIMENSIONS, "distance": "Cosine"}})
        response.raise_for_status()
        points = index.qdrant_points()
        for start in range(0, len(points), 100):
            response = await client.put(url + "/points?wait=true", json={"points": points[start:start + 100]})
            response.raise_for_status()
