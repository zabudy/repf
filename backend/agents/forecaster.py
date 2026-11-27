"""Forecaster: строит прогноз цены.

ВНИМАНИЕ: сейчас это ЗАГЛУШКА с условными коэффициентами, чтобы проверить работу
всей цепочки агентов. Цифры НЕ отражают реальный рынок. На этапах 6–8 она
заменяется ансамблем ML-моделей, обученных на реальных данных.
"""
from datetime import date

from backend.agents.schemas import PropertyInput, RawForecast

BASE_PRICE_M2 = 150_000  # условная база, ₽/м²
MONTHLY_GROWTH = 0.004   # условный рост цен в месяц

HOUSE_TYPE_K = {"panel": 0.95, "brick": 1.05, "monolith": 1.10, "block": 0.90, "wood": 0.75}
CONDITION_K = {"rough": 0.85, "needs_repair": 0.90, "cosmetic": 1.00, "euro": 1.10, "designer": 1.20}


class BaselineForecaster:
    name = "Forecaster"
    model_name = "baseline-stub-v0"

    def run(self, p: PropertyInput) -> RawForecast:
        k = HOUSE_TYPE_K.get(p.house_type or "", 1.0) * CONDITION_K.get(p.condition or "", 1.0)

        if p.floor == 1 or p.floor == p.floors_total:
            k *= 0.95  # первый и последний этажи дешевле

        if p.build_year:
            age = max(date.today().year - p.build_year, 0)
            k *= max(1.0 - 0.002 * age, 0.8)

        price_m2 = BASE_PRICE_M2 * k * (1 + MONTHLY_GROWTH) ** p.horizon_months
        price = price_m2 * p.area_total
        width = 0.08 + 0.01 * p.horizon_months  # чем дальше горизонт, тем шире интервал
        return RawForecast(
            price=round(price),
            interval_low=round(price * (1 - width)),
            interval_high=round(price * (1 + width)),
            model_name=self.model_name,
        )
