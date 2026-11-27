"""Create domain-specific prepared artifacts from the input reference."""
import asyncio

import numpy as np

from backend.agents.base import Agent, Artifacts, Trace
from backend.core.forecast import ForecastState, SeriesInput, TabularInput
from backend.tabular.housing import prepare_market
from backend.timeseries.preprocessing import diagnostics, feature_vector, prepare


def analyse(data: SeriesInput | TabularInput, statistical: bool = True):
    if isinstance(data, SeriesInput):
        series = prepare(data)
        indicators = diagnostics(series.values, series.period, statistical)
        metadata = {"task_type": "timeseries", "frequency": series.frequency, "period": series.period,
                    "missing_rate": series.missing_rate, **indicators}
        return series, metadata, feature_vector(series.values)
    market = prepare_market(data)
    missing = sum(getattr(row, field) is None for row in market.sales for field in
                  ("build_year", "house_type", "condition")) / (len(market.sales) * 7)
    district = market.apartment.district.casefold()
    numeric = market.apartment
    base_features = feature_vector(market.index.values)
    comparisons = [row for row in market.sales if row.district.casefold() == district]
    sample = comparisons if comparisons else market.sales
    locality = [numeric.area_total / 100, numeric.floor / numeric.floors_total,
                (numeric.build_year or 1985) / 2100, missing,
                len(comparisons) / len(market.sales),
                float(np.std([row.price_rub / row.area_total for row in sample])) /
                max(float(np.mean([row.price_rub / row.area_total for row in sample])), 1e-12)]
    meta = {"task_type": "tabular", "problem_type": "regression", "domain": "housing", "num_features": 7,
            "target_column": "price_rub", "count": len(market.sales), "missing_rate": missing,
            "district": market.apartment.district, "district_comparables": len(comparisons),
            "demo": market.demo}
    return market, meta, np.r_[base_features, locality]


class DataAnalyst(Agent):
    def __init__(self, artifacts: Artifacts, trace: Trace, statistical: bool):
        self.artifacts, self.trace, self.statistical = artifacts, trace, statistical

    async def process(self, state: ForecastState, parent_id: str) -> None:
        async with self.trace.step("DataAnalyst", parent_id, task_type=state.task_type) as (_, output):
            data = self.artifacts.get(state.input_id)
            prepared, state.metadata, features = await asyncio.to_thread(analyse, data, self.statistical)
            state.prepared_id = self.artifacts.put((prepared, features))
            output.update(state.metadata)
