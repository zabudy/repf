"""Task state contains references to in-process artifacts and small summaries."""
from abc import ABC, abstractmethod
from asyncio import CancelledError
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from backend.core.forecast import ForecastState


class Artifacts:
    def __init__(self):
        self._data: dict[str, Any] = {}

    def put(self, value: Any) -> str:
        artifact_id = str(uuid4())
        self._data[artifact_id] = value
        return artifact_id

    def get(self, artifact_id: str | None) -> Any:
        if artifact_id is None:
            raise RuntimeError("An artifact has not been produced")
        return self._data[artifact_id]


class TraceSink(Protocol):
    async def span(self, entry: dict) -> None: ...
    async def llm_call(self, span_id: str, entry: dict) -> None: ...


class Trace:
    def __init__(self, sink: TraceSink | None = None):
        self.sink = sink
        self.tokens = 0
        self.calls = 0
        self.estimated_calls = 0

    @asynccontextmanager
    async def step(self, agent: str, parent_id: str | None = None, **inputs):
        span_id, start = str(uuid4()), perf_counter()
        entry = {"id": span_id, "parent_id": parent_id, "agent": agent, "status": "running",
                 "details": {"input": inputs}}
        if self.sink:
            await self.sink.span(entry)
        output: dict = {}
        try:
            yield span_id, output
            entry["status"] = "ok"
        except CancelledError:
            entry["status"] = "cancelled"
            raise
        except Exception as error:
            output["error_type"] = type(error).__name__
            entry["status"] = "error"
            raise
        finally:
            entry["details"].update(output=output, latency_ms=round((perf_counter() - start) * 1000, 3))
            if self.sink:
                await self.sink.span(entry)

    async def llm_call(self, span_id: str, entry: dict) -> None:
        self.tokens += (entry.get("input_tokens") or 0) + (entry.get("output_tokens") or 0)
        self.calls += 1
        self.estimated_calls += bool(entry.get("estimated_tokens"))
        if self.sink:
            await self.sink.llm_call(span_id, entry)


class Agent(ABC):
    @abstractmethod
    async def process(self, state: ForecastState, parent_id: str) -> None: ...
