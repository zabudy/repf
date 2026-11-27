"""Прогнать несколько примеров через цепочку агентов: python -m scripts.demo_agents"""
from backend.agents.schemas import PropertyInput
from backend.agents.supervisor import Supervisor

EXAMPLES = {
    "Обычная квартира": PropertyInput(
        area_total=54, floor=5, floors_total=9, build_year=1985,
        house_type="panel", condition="cosmetic", horizon_months=6,
    ),
    "Мало данных, дальний горизонт": PropertyInput(
        area_total=250, floor=1, floors_total=3, horizon_months=12,
    ),
    "Ошибка во вводе": PropertyInput(
        area_total=45, floor=12, floors_total=9, build_year=2010,
        house_type="panel", condition="euro", horizon_months=3,
    ),
}

if __name__ == "__main__":
    sup = Supervisor()
    for title, prop in EXAMPLES.items():
        res = sup.run(prop)
        print("=" * 70)
        print(title)
        for step in res.trace:
            print(f"  [{step.status:7}] {step.agent:10} {step.message}")
        print("  ->", res.explanation)
