import numpy as np
import pandas as pd
import pytest

from backend.agents.forecaster import ModelForecaster
from backend.agents.schemas import PropertyInput
from backend.agents.supervisor import Supervisor
from backend.data.loader import clean, haversine_km
from backend.data.synthetic import make_synthetic
from backend.features.build import FEATURES, build_features
from backend.models.hedonic import train_pipeline
from backend.models.market_index import fit_trend, monthly_index, trend_value


@pytest.fixture(scope="module")
def data():
    df, _ = clean(make_synthetic(8000, seed=1))
    return df


def test_haversine_known_distance():
    # Казань — Москва по прямой около 720 км
    assert 700 < haversine_km(55.79, 49.12, 55.75, 37.62) < 740


def test_clean_removes_garbage():
    raw = make_synthetic(500, seed=2)
    bad = raw.iloc[:5].copy()
    bad.loc[bad.index[0], "area"] = 5            # слишком маленькая
    bad.loc[bad.index[1], "level"] = 99          # этаж выше этажности
    bad.loc[bad.index[2], "price"] = 1000        # цена за м² абсурдна
    bad.loc[bad.index[3], "object_type"] = 2     # новостройка
    bad.loc[bad.index[4], "kitchen_area"] = 999  # кухня больше квартиры -> NaN, строка остаётся
    df, stats = clean(pd.concat([raw.iloc[5:], bad]))
    assert stats["input"] == 500
    assert len(df) == 496  # 500 минус 4 испорченные
    assert df["kitchen_area"].isna().sum() >= 1
    assert (df["rooms"] >= 0).all()


def test_features_shape(data):
    X = build_features(data.head(10))
    assert list(X.columns) == FEATURES and len(X) == 10


def test_index_and_trend(data):
    idx = monthly_index(data)
    slope, _ = fit_trend(idx)
    assert 0.001 < slope < 0.01  # в синтетике заложен рост ≈ 0.4 % в месяц
    assert trend_value(fit_trend(idx), idx.index[-1] + 3) > idx.iloc[-1] - 0.05


@pytest.fixture(scope="module")
def trained(data):
    return train_pipeline(data, params=dict(n_estimators=120))


def test_model_beats_baseline(trained):
    _, m = trained
    assert m["mape_model"] < m["mape_baseline_median"]
    assert m["mape_model"] < 0.15
    assert 0.5 < m["interval_coverage_80"] <= 1.0


def test_model_forecaster_and_supervisor(trained):
    bundle, _ = trained
    f = ModelForecaster(bundle)
    near = f.run(PropertyInput(area_total=50, floor=4, floors_total=9, house_type="brick",
                               geo_lat=55.80, geo_lon=49.11, horizon_months=3))
    far = f.run(PropertyInput(area_total=50, floor=4, floors_total=9, house_type="brick",
                              geo_lat=55.65, geo_lon=49.40, horizon_months=3))
    assert near.price > far.price  # ближе к центру дороже
    assert near.interval_low < near.price < near.interval_high

    res = Supervisor(forecaster=f).run(
        PropertyInput(area_total=50, floor=4, floors_total=9, house_type="brick", horizon_months=6)
    )
    assert res.price and res.model_name == "lgbm-hedonic+trend"
    assert "Ограничения модели" in res.explanation
