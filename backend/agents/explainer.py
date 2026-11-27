"""Explainer: человекочитаемое пояснение.

Сейчас — по шаблону. На этапе 8 к нему подключится LLM (Ollama) и SHAP-факторы.
"""
from backend.agents.schemas import Critique, PropertyInput, RawForecast


def _rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " ₽"


class Explainer:
    name = "Explainer"

    def run(self, p: PropertyInput, f: RawForecast, c: Critique, needs_review: bool) -> str:
        parts = [
            f"Прогноз цены через {p.horizon_months} мес.: {_rub(f.price)} "
            f"(интервал {_rub(f.interval_low)} – {_rub(f.interval_high)}), "
            f"уверенность {c.confidence:.0%}.",
            f"Объект: {p.area_total:g} м², {p.floor}/{p.floors_total} этаж"
            + (f", {p.house_type}" if p.house_type else "")
            + (f", {p.build_year} года постройки" if p.build_year else "")
            + ".",
        ]
        if c.notes:
            parts.append("Замечания: " + "; ".join(c.notes) + ".")
        if f.notes:
            parts.append("Ограничения модели: " + "; ".join(f.notes) + ".")
        if needs_review:
            parts.append("Рекомендуется проверка экспертом.")
        return " ".join(parts)
