"""Critic: оценивает правдоподобие прогноза и выставляет уверенность 0..1."""
from backend.agents.schemas import Critique, PropertyInput, RawForecast, ValidationReport
from backend.core import constants as C


class Critic:
    name = "Critic"

    def run(self, p: PropertyInput, f: RawForecast, v: ValidationReport) -> Critique:
        confidence = 1.0
        notes: list[str] = []

        # 1. Неполные входные данные
        if v.warnings:
            confidence -= 0.08 * len(v.warnings)
            notes.append(f"Неполные данные: {len(v.warnings)} предупр.")

        # 2. Цена за м² в правдоподобных границах
        price_m2 = f.price / p.area_total
        if not C.PRICE_M2_MIN <= price_m2 <= C.PRICE_M2_MAX:
            confidence -= 0.4
            notes.append(f"Цена за м² ({price_m2:,.0f} ₽) вне правдоподобного диапазона")

        # 3. Ширина интервала
        rel_width = (f.interval_high - f.interval_low) / (2 * f.price)
        if rel_width > 0.15:
            confidence -= 0.1
            notes.append(f"Широкий интервал (±{rel_width:.0%})")

        # 4. Дальний горизонт
        confidence -= 0.02 * p.horizon_months
        if p.horizon_months > 9:
            notes.append("Дальний горизонт прогноза")

        # 5. Нетипичные объекты
        if p.area_total < 20 or p.area_total > 200:
            confidence -= 0.1
            notes.append("Нетипичная площадь")
        if p.build_year and p.build_year < 1930:
            confidence -= 0.1
            notes.append("Очень старый дом")

        return Critique(confidence=round(min(max(confidence, 0.0), 1.0), 2), notes=notes)
