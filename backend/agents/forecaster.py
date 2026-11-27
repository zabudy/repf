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


# ----------------------------------------------------------------------------
# Настоящий прогнозист (этап 6): LightGBM + индекс рынка
# ----------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.core import constants as C  # noqa: E402
from backend.core.config import get_settings  # noqa: E402
from backend.features.build import build_features  # noqa: E402
from backend.models.hedonic import HORIZON_WIDEN, load_bundle  # noqa: E402
from backend.models.market_index import trend_value  # noqa: E402


class ModelForecaster:
    name = "Forecaster"
    model_name = "lgbm-hedonic+trend"

    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, path: str | Path) -> "ModelForecaster":
        return cls(load_bundle(path))

    def run(self, p: PropertyInput) -> RawForecast:
        lat0, lon0 = self.b["geo_median"]
        row = pd.DataFrame([{
            "area": p.area_total,
            "kitchen_area": p.kitchen_area if p.kitchen_area else np.nan,
            "rooms": p.rooms if p.rooms is not None else np.nan,
            "level": p.floor,
            "levels": p.floors_total,
            "building_type": C.BUILDING_TYPE_CODE.get(p.house_type or "", 0),
            "geo_lat": p.geo_lat if p.geo_lat is not None else lat0,
            "geo_lon": p.geo_lon if p.geo_lon is not None else lon0,
        }])
        month = self.b["last_month"] + p.horizon_months
        log_m2 = float(self.b["model"].predict(build_features(row))[0]) + float(
            trend_value(self.b["trend"], month)
        )
        widen = HORIZON_WIDEN * p.horizon_months
        price = float(np.exp(log_m2) * p.area_total)
        low = float(np.exp(log_m2 + self.b["q_low"] - widen) * p.area_total)
        high = float(np.exp(log_m2 + self.b["q_high"] + widen) * p.area_total)

        notes = ["модель не учитывает состояние ремонта и год постройки (их нет в обучающих данных)"]
        if p.geo_lat is None or p.geo_lon is None:
            notes.append("координаты не заданы — использован центр выборки, точность ниже")
        return RawForecast(
            price=round(price), interval_low=round(low), interval_high=round(high),
            model_name=self.model_name, notes=notes,
        )


def load_default_forecaster():
    """ML-модель, если файл обучен; иначе заглушка."""
    path = Path(get_settings().MODEL_PATH)
    if path.exists():
        try:
            return ModelForecaster.load(path)
        except Exception:  # повреждённый файл не должен ронять сервис
            pass
    return BaselineForecaster()
