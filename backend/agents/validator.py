"""Validator: проверяет входные данные до прогноза."""
from datetime import date

from backend.agents.schemas import PropertyInput, ValidationReport
from backend.core import constants as C


class Validator:
    name = "Validator"

    def run(self, p: PropertyInput) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        if p.region not in C.SUPPORTED_REGIONS:
            errors.append(f"Регион «{p.region}» не поддерживается (доступно: {', '.join(C.SUPPORTED_REGIONS)})")
        if not C.MIN_AREA_M2 <= p.area_total <= C.MAX_AREA_M2:
            errors.append(f"Площадь {p.area_total} м² вне диапазона {C.MIN_AREA_M2}–{C.MAX_AREA_M2}")
        if p.floors_total < 1 or p.floors_total > C.MAX_FLOORS:
            errors.append(f"Этажность дома {p.floors_total} некорректна")
        if p.floor < 1 or p.floor > max(p.floors_total, 1):
            errors.append(f"Этаж {p.floor} не может быть больше этажности ({p.floors_total}) или меньше 1")
        if not C.MIN_HORIZON_MONTHS <= p.horizon_months <= C.MAX_HORIZON_MONTHS:
            errors.append(f"Горизонт {p.horizon_months} мес. вне диапазона {C.MIN_HORIZON_MONTHS}–{C.MAX_HORIZON_MONTHS}")

        if p.build_year is None:
            warnings.append("Не указан год постройки")
        elif not C.MIN_BUILD_YEAR <= p.build_year <= date.today().year + 5:
            errors.append(f"Год постройки {p.build_year} некорректен")

        if p.house_type is None:
            warnings.append("Не указан тип дома")
        elif p.house_type not in C.HOUSE_TYPES:
            errors.append(f"Неизвестный тип дома «{p.house_type}» (доступно: {', '.join(C.HOUSE_TYPES)})")

        if p.condition is None:
            warnings.append("Не указано состояние квартиры")
        elif p.condition not in C.CONDITIONS:
            errors.append(f"Неизвестное состояние «{p.condition}» (доступно: {', '.join(C.CONDITIONS)})")

        return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
