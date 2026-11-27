"""Choose a registered model using diagnostics, case evidence and function calls."""
from backend.agents.base import Agent, Artifacts, Trace
from backend.core.forecast import ForecastState, ToolChoice
from backend.core.llm.manager import LLMManager
from backend.core.llm.tools import available, validate
from backend.retrieval.hybrid import HybridRetriever


def default_choice(allowed: list[str], meta: dict) -> ToolChoice:
    if meta["task_type"] == "tabular":
        preferred = "fit_ridge"
    elif meta.get("seasonality", 0) > .45:
        preferred = "fit_harmonic"
    elif meta.get("autocorrelation", 0) > .6 and abs(meta.get("trend", 0)) < 1:
        preferred = "fit_autoregression"
    else:
        preferred = "fit_holt"
    selected = preferred if preferred in allowed else allowed[0]
    return validate(ToolChoice(name=selected), meta["task_type"])


def similar_choice(cases: list[dict], allowed: list[str], task_type, default: ToolChoice) -> ToolChoice:
    scores = {name: sum(row["rrf_score"] for row in cases if row["model"] == name)
              for name in allowed}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return default
    row = next(row for row in cases if row["model"] == best)
    return validate(ToolChoice(name=best, arguments=row["arguments"]), task_type)


class ModelSelector(Agent):
    def __init__(self, artifacts: Artifacts, trace: Trace, retriever: HybridRetriever,
                 llm: LLMManager, forced: str | None = None):
        self.artifacts, self.trace, self.retriever, self.llm = artifacts, trace, retriever, llm
        self.forced = forced

    async def process(self, state: ForecastState, parent_id: str) -> None:
        allowed = available(state.task_type, state.attempted)
        async with self.trace.step("ModelSelector", parent_id, mode=state.mode, allowed=allowed) as (span, output):
            if self.forced and not state.attempted:
                state.choice = validate(ToolChoice(name=self.forced), state.task_type)
                output.update(source="human")
            elif state.mode == "baseline":
                state.choice = (ToolChoice(name="seasonal_naive") if state.task_type == "timeseries"
                                else validate(ToolChoice(name="fit_ridge"), state.task_type))
                output.update(source="baseline")
            else:
                _, features = self.artifacts.get(state.prepared_id)
                if state.iteration == 1:
                    async with self.trace.step("HybridRetriever", span) as (_, retrieved):
                        state.retrieved, backend = await self.retriever.retrieve(state.metadata, features)
                        retrieved.update(backend=backend, cases=[row["id"] for row in state.retrieved])
                fallback = default_choice(allowed, state.metadata)
                case_choice = similar_choice(state.retrieved, allowed, state.task_type, fallback)
                if state.mode == "+rag":
                    state.choice = case_choice
                    output.update(source="rag" if state.retrieved else "fallback")
                else:
                    async def log(entry: dict) -> None:
                        await self.trace.llm_call(span, entry)
                    state.choice, source = await self.llm.select(
                        state.task_type, allowed,
                        {"horizon": state.horizon, "diagnostics": state.metadata,
                         "cases": state.retrieved, "failed_tools": state.attempted,
                         "previous_validation": state.diagnostics},
                        case_choice, log,
                    )
                    output.update(source=source)
            output["tool"] = state.choice.model_dump()
