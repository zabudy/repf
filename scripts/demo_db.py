"""Записать и прочитать тестовый объект: python -m scripts.demo_db"""
import asyncio

from sqlalchemy import select

from backend.db.models import Forecast, Property
from backend.db.session import SessionLocal, init_db


async def main() -> None:
    await init_db()
    async with SessionLocal() as s:
        prop = Property(
            region="Kazan", district="Вахитовский", area_total=54.0,
            floor=5, floors_total=9, build_year=1985, house_type="panel",
            condition="cosmetic", price_rub=6_200_000, source="demo",
        )
        prop.forecasts.append(
            Forecast(horizon_months=6, predicted_price=6_500_000,
                     interval_low=6_100_000, interval_high=6_900_000,
                     confidence=0.72, model_name="demo")
        )
        s.add(prop)
        await s.commit()

        rows = (await s.execute(select(Property))).scalars().all()
        print(f"Объектов в базе: {len(rows)}")
        last = rows[-1]
        print("Последний:", last.region, last.district, last.area_total, "м²", last.price_rub, "₽")


if __name__ == "__main__":
    asyncio.run(main())
