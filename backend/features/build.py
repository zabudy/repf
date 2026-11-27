"""Признаки для модели."""
import numpy as np
import pandas as pd

from backend.core import constants as C
from backend.data.loader import haversine_km

FEATURES = [
    "area", "kitchen_area", "rooms", "level", "levels",
    "is_first", "is_last", "level_ratio",
    "building_type", "geo_lat", "geo_lon", "dist_center_km",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Из колонок area, kitchen_area, rooms, level, levels, building_type, geo_lat, geo_lon."""
    X = pd.DataFrame(index=df.index)
    X["area"] = df["area"].astype(float)
    X["kitchen_area"] = df["kitchen_area"].astype(float)
    X["rooms"] = df["rooms"].astype(float)
    X["level"] = df["level"].astype(float)
    X["levels"] = df["levels"].astype(float)
    X["is_first"] = (df["level"] == 1).astype(int)
    X["is_last"] = (df["level"] == df["levels"]).astype(int)
    X["level_ratio"] = df["level"] / df["levels"]
    X["building_type"] = pd.Categorical(df["building_type"].astype(int), categories=range(6))
    X["geo_lat"] = df["geo_lat"].astype(float)
    X["geo_lon"] = df["geo_lon"].astype(float)
    X["dist_center_km"] = np.asarray(haversine_km(df["geo_lat"], df["geo_lon"], *C.KAZAN_CENTER))
    return X[FEATURES]
