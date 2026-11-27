from datetime import date, timedelta

import numpy as np
import pytest

from backend.core.forecast import SeriesInput
from backend.core.metrics import accuracy, residual_diagnostics
from backend.timeseries.preprocessing import diagnostics, feature_vector, prepare


def test_resample_duplicates_and_forward_fill():
    day = date(2022, 10, 1)
    rows = [{"date": day + timedelta(days=i), "value": i + 10} for i in range(17) if i not in (3, 8)]
    rows.append({"date": day, "value": 16})
    series = prepare(SeriesInput(observations=list(reversed(rows)), frequency="D"))
    assert series.values[:5].tolist() == [13, 11, 12, 12, 14]
    assert len(series.index) == 17
    assert series.missing_rate == pytest.approx(2 / 17)
    assert np.isfinite(feature_vector(series.values)).all()
    eda = diagnostics(series.values, 7)
    assert eda["count"] == 17
    pytest.importorskip("statsmodels")
    assert 0 <= eda["adf_pvalue"] <= 1
    assert 0 <= eda["stl_strength"] <= 1


def test_constant_and_zero_diagnostics():
    actual = np.zeros(9)
    scores = accuracy(actual, np.ones(9), actual, 7)
    assert scores == {"mae": 1., "rmse": 1., "mape": None, "mase": None, "wape": None}
    assert residual_diagnostics(actual)["acf"] == [0, 0, 0]


def test_negative_prices_are_rejected():
    rows = [{"date": date(2025, 1, 1) + timedelta(days=i), "value": i - 2} for i in range(15)]
    with pytest.raises(ValueError, match="отрицательном"):
        prepare(SeriesInput(observations=rows, nonnegative=True))


def test_residual_pvalues_when_statistical_backend_is_installed():
    pytest.importorskip("statsmodels")
    result = residual_diagnostics(np.random.default_rng(12).normal(0, 1, 160))
    assert len(result["acf"]) == len(result["pacf"]) == 10
    assert 0 <= result["ljung_box_pvalue"] <= 1
    assert 0 <= result["shapiro_pvalue"] <= 1
