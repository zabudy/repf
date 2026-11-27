"""LLM-as-a-Judge with a deterministic validation-based fallback."""
from backend.agents.base import Agent, Artifacts, Trace
from backend.core.forecast import ForecastState
from backend.core.llm.manager import Critique, LLMManager


class Critic(Agent):
    def __init__(self, artifacts: Artifacts, trace: Trace, llm: LLMManager):
        self.artifacts, self.trace, self.llm = artifacts, trace, llm

    async def process(self, state: ForecastState, parent_id: str) -> None:
        async with self.trace.step("Critic", parent_id) as (span, output):
            result = self.artifacts.get(state.result_id)
            mase, wape = result["metrics"]["mase"], result["metrics"]["wape"]
            if state.task_type == "timeseries":
                estimated = max(.2, 1 - .22 * mase) if mase is not None else .75
            else:
                estimated = max(.2, 1 - .012 * wape) if wape is not None else .6
            reasons = state.diagnostics["reasons"]
            fallback = Critique(score=min(estimated, .48) if reasons else estimated,
                                comment="; ".join(reasons) if reasons else
                                "Формальные проверки пройдены. Оценка основана на отложенных наблюдениях.")
            async def log(entry: dict) -> None:
                await self.trace.llm_call(span, entry)
            critique = await self.llm.critique(
                {"task_type": state.task_type, "model": result["model"], "metrics": result["metrics"],
                 "validation": state.diagnostics, "data": state.metadata}, fallback, log,
            )
            state.critic_score, state.critic_comment = critique.score, critique.comment
            output.update(critique.model_dump())
