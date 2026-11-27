"""Supervisor: управляет цепочкой Validator → Forecaster → Critic → Explainer."""
from backend.agents.critic import Critic
from backend.agents.explainer import Explainer
from backend.agents.forecaster import BaselineForecaster
from backend.agents.schemas import AgentStep, ForecastResult, PropertyInput
from backend.agents.validator import Validator
from backend.core.config import get_settings


class Supervisor:
    def __init__(self, forecaster=None, confidence_threshold: float | None = None):
        self.validator = Validator()
        self.forecaster = forecaster or BaselineForecaster()
        self.critic = Critic()
        self.explainer = Explainer()
        self.threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_settings().CONFIDENCE_THRESHOLD
        )

    def run(self, p: PropertyInput) -> ForecastResult:
        trace: list[AgentStep] = []

        # 1. Validator
        report = self.validator.run(p)
        if not report.ok:
            trace.append(AgentStep(agent="Validator", status="error", message="; ".join(report.errors)))
            return ForecastResult(
                needs_review=False,
                explanation="Прогноз не построен: " + "; ".join(report.errors) + ".",
                trace=trace,
            )
        trace.append(
            AgentStep(
                agent="Validator",
                status="warning" if report.warnings else "ok",
                message="; ".join(report.warnings) or "Данные в порядке",
            )
        )

        # 2. Forecaster
        raw = self.forecaster.run(p)
        trace.append(AgentStep(agent="Forecaster", status="ok", message=f"{raw.model_name}: {raw.price:,.0f} ₽"))

        # 3. Critic
        critique = self.critic.run(p, raw, report)
        trace.append(
            AgentStep(
                agent="Critic",
                status="warning" if critique.notes else "ok",
                message=f"Уверенность {critique.confidence:.2f}" + (f"; {'; '.join(critique.notes)}" if critique.notes else ""),
            )
        )

        # 4. Решение: нужна ли проверка экспертом
        needs_review = critique.confidence < self.threshold
        trace.append(
            AgentStep(
                agent="Supervisor",
                status="warning" if needs_review else "ok",
                message="Отправить эксперту" if needs_review else "Результат принят",
            )
        )

        # 5. Explainer
        text = self.explainer.run(p, raw, critique, needs_review)
        trace.append(AgentStep(agent="Explainer", status="ok", message="Пояснение сформировано"))

        return ForecastResult(
            price=raw.price,
            price_per_m2=round(raw.price / p.area_total),
            interval_low=raw.interval_low,
            interval_high=raw.interval_high,
            confidence=critique.confidence,
            needs_review=needs_review,
            model_name=raw.model_name,
            explanation=text,
            trace=trace,
        )
