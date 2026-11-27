"""Загрузка и очистка датасета объявлений (формат Russia_Real_Estate_2018_2021)."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core import constants as C
from backend.core.exceptions import DataValidationError

log = logging.getLogger(__name__)

REQUIRED = [
    "price", "date", "geo_lat", "geo_lon", "building_type", "object_type",
    "level", "levels", "rooms", "area", "kitchen_area",
]
NUMERIC = [c for c in REQUIRED if c != "date"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние между точками на Земле, км (работает с числами и колонками)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def read_kazan_raw(path: str | Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Прочитать большой CSV кусками и оставить только прямоугольник вокруг Казани."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise DataValidationError(f"В файле нет колонок {missing}. Есть: {header}")

    lat_lo, lat_hi = C.KAZAN_BBOX["lat"]
    lon_lo, lon_hi = C.KAZAN_BBOX["lon"]
    parts, total = [], 0
    for chunk in pd.read_csv(path, usecols=REQUIRED, chunksize=chunksize):
        total += len(chunk)
        lat = pd.to_numeric(chunk["geo_lat"], errors="coerce")
        lon = pd.to_numeric(chunk["geo_lon"], errors="coerce")
        parts.append(chunk[lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi)])
        log.info("Прочитано строк: %s", f"{total:,}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REQUIRED)
    log.info("В прямоугольнике Казани: %s из %s", f"{len(df):,}", f"{total:,}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка: типы, фильтры правдоподобия, дубли. Возвращает (данные, статистика)."""
    stats = {"input": len(df)}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    must = [c for c in REQUIRED if c != "kitchen_area"]
    df = df.dropna(subset=must)
    stats["after_dropna"] = len(df)

    df = df[df["object_type"] == 1]  # только вторичный рынок
    stats["secondary_only"] = len(df)

    df = df[df["area"].between(C.MIN_AREA_M2, C.MAX_AREA_CLEAN_M2)]
    df = df[(df["level"] >= 1) & (df["levels"] >= 1) & (df["level"] <= df["levels"]) & (df["levels"] <= C.MAX_FLOORS)]
    df = df[df["rooms"].between(-1, 10)]
    df["rooms"] = df["rooms"].clip(lower=0)  # -1 (студия) -> 0
    df = df[df["building_type"].between(0, 5)]
    stats["after_basic_filters"] = len(df)

    df["price_m2"] = df["price"] / df["area"]
    df = df[df["price_m2"].between(C.PRICE_M2_MIN, C.PRICE_M2_MAX)]
    stats["after_price_filter"] = len(df)

    bad_kitchen = (df["kitchen_area"] <= 0) | (df["kitchen_area"] >= df["area"])
    df.loc[bad_kitchen, "kitchen_area"] = np.nan

    df = df.drop_duplicates(subset=["price", "date", "geo_lat", "geo_lon", "area", "level", "levels"])
    stats["after_dedup"] = len(df)

    df["dist_center_km"] = haversine_km(df["geo_lat"], df["geo_lon"], *C.KAZAN_CENTER)
    df = df[df["dist_center_km"] <= C.KAZAN_MAX_DIST_KM]
    stats["output"] = len(df)

    df["month"] = df["date"].dt.year * 12 + df["date"].dt.month
    df["building_type"] = df["building_type"].astype(int)
    return df.reset_index(drop=True), stats


def month_label(month: int) -> str:
    """Номер месяца (год*12+месяц) -> 'ГГГГ-ММ'."""
    y, m = divmod(month - 1, 12)
    return f"{y}-{m + 1:02d}"
