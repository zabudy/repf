"""ДЕМО-данные в формате реального датасета. Это НЕ реальный рынок —
нужны, чтобы проверять код без скачивания 400 МБ."""
import numpy as np
import pandas as pd

from backend.core import constants as C


def make_synthetic(n: int = 20_000, seed: int = 0, start: str = "2018-02-01", months: int = 39) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, months, n)
    day = rng.integers(0, 28, n)
    dates = pd.to_datetime(start) + pd.to_timedelta(t * 30 + day, unit="D")

    lat = C.KAZAN_CENTER[0] + rng.normal(0, 0.05, n)
    lon = C.KAZAN_CENTER[1] + rng.normal(0, 0.09, n)
    dist = np.sqrt(((lat - C.KAZAN_CENTER[0]) * 111) ** 2 + ((lon - C.KAZAN_CENTER[1]) * 62) ** 2)

    levels = rng.choice([5, 9, 12, 16, 25], n)
    level = (rng.random(n) * levels).astype(int) + 1
    rooms = rng.choice([-1, 1, 2, 3, 4], n, p=[0.05, 0.35, 0.35, 0.2, 0.05])
    area = np.clip(rng.normal(30 + 15 * np.maximum(rooms, 0), 8, n), 14, 200)
    kitchen = np.clip(area * rng.uniform(0.1, 0.25, n), 4, 40)
    btype = rng.choice([1, 2, 3, 4], n, p=[0.45, 0.2, 0.3, 0.05])

    k_type = np.array([1.0, 0.95, 1.10, 1.0, 0.92])[btype]
    k_floor = np.where((level == 1) | (level == levels), 0.94, 1.0)
    price_m2 = (
        85_000 * np.exp(0.004 * t) * k_type * k_floor * np.exp(-0.012 * dist)
        * (1 + 0.0004 * (area - 45)) * np.exp(rng.normal(0, 0.10, n))
    )
    return pd.DataFrame({
        "price": (price_m2 * area).round(),
        "date": dates.strftime("%Y-%m-%d"),
        "geo_lat": lat, "geo_lon": lon,
        "building_type": btype, "object_type": 1,
        "level": level, "levels": levels, "rooms": rooms,
        "area": area.round(1), "kitchen_area": kitchen.round(1),
    })
