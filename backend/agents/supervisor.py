"""Run the bounded sequence of agents and surface reviewable outcomes."""
from dataclasses import dataclass
from time import perf_counter

from backend.agents.base import Artifacts, Trace, TraceSink
from backend.agents.critic import Critic
from backend.agents.data_analyst import DataAnalyst
from backend.agents.forecaster import Forecaster
from backend.agents.model_selector import ModelSelector
from backend.agents.validator import Validator
from backend.core.config import Settings
from backend.core.forecast import ForecastRequest, ForecastState
from backend.core.llm.manager import LLMManager
from backend.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class Outcome:
    state: ForecastState
    result: dict
    needs_review: bool


class Supervisor:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: LLMManager | None = None):
        self.settings, self.retriever = settings, retriever
        self.llm = llm or LLMManager(settings)

    async def run(self, task_id: str, request: ForecastRequest, sink: TraceSink | None = None,
                  review: dict | None = None) -> Outcome:
        started = perf_counter()
        artifacts, trace = Artifacts(), Trace(sink)
        state = ForecastState(task_id=task_id, input_id=artifacts.put(request.data),
                              task_type=request.data.task_type, horizon=request.horizon,
                              target_column=getattr(request.data, "target_column", None), mode=request.mode,
                              hitl_resolution=review)
        forced_model = (review or {}).get("model")
        selector = ModelSelector(artifacts, trace, self.retriever, self.llm, forced_model)
        statistical = self.settings.STATISTICAL_DIAGNOSTICS
        forecaster = Forecaster(artifacts, trace, statistical)
        validator = Validator(artifacts, trace)
        critic = Critic(artifacts, trace, self.llm)
        limit = self.settings.MAX_ITERATIONS if request.mode == "+agents" and not forced_model else 1
        async with trace.step("Supervisor", horizon=request.horizon, mode=request.mode) as (root, output):
            await DataAnalyst(artifacts, trace, statistical).process(state, root)
            passed = False
            candidates = []
            for iteration in range(1, limit + 1):
                state.iteration = iteration
                async with trace.step("Iteration", root, iteration=iteration) as (parent, attempt):
                    await selector.process(state, parent)
                    await forecaster.process(state, parent)
                    await validator.process(state, parent)
                    if request.mode == "+agents":
                        await critic.process(state, parent)
                    passed = state.validation_passed and (state.critic_score is None or
                                                          state.critic_score >= self.settings.CONFIDENCE_THRESHOLD)
                    assert state.choice is not None
                    state.attempted.append(state.choice.name)
                    result = artifacts.get(state.result_id)
                    candidates.append((state.model_copy(deep=True), result))
                    attempt.update(passed=passed, model=state.choice.name)
                    if passed:
                        break
            selected_state, result = candidates[-1] if passed else min(
                candidates, key=lambda item: (not item[0].validation_passed, item[1]["metrics"]["mae"]))
            review_required = not passed and request.mode == "+agents" and self.settings.HITL_ENABLED
            result.update(
                data_summary=selected_state.metadata, validation=selected_state.diagnostics,
                confidence=selected_state.critic_score, explanation=selected_state.critic_comment,
                cases=state.retrieved, needs_review=review_required,
                trajectory={"iterations": state.iteration, "llm_calls": trace.calls,
                            "tokens": trace.tokens, "estimated_calls": trace.estimated_calls,
                            "latency_ms": round((perf_counter() - started) * 1000, 3)},
            )
            selected_state.iteration = state.iteration
            selected_state.attempted = state.attempted
            output.update(model=result["model"]["name"], needs_review=review_required,
                          trajectory=result["trajectory"])
        return Outcome(selected_state, result, review_required)
