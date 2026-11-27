"""Paired temporal experiments over disjoint benchmark series."""
import asyncio
import csv
from itertools import combinations, product
from math import erfc, sqrt
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
from pydantic import Field
from typing import Literal

from backend.agents.supervisor import Supervisor
from backend.core.config import Settings
from backend.core.forecast import ForecastRequest, InputModel, Mode
from backend.core.metrics import accuracy
from backend.data.examples import PERIODS, synthetic_series
from backend.retrieval.hybrid import HybridRetriever, load_cases

MODES = ["baseline", "+rag", "+agents"]


class EvalParameters(InputModel):
    num_series: int = Field(default=20, ge=1, le=100)
    horizons: list[int] = Field(default_factory=lambda: [10, 20, 50], min_length=1, max_length=5)
    data_fractions: list[float] = Field(default_factory=lambda: [.1, .2, .5, 1.], min_length=1, max_length=5)
    modes: list[Mode] = Field(default_factory=lambda: list(MODES), min_length=1, max_length=3)
    dataset: Literal["synthetic", "macro", "tatarstan"] = "synthetic"
    seed: int = 5003
    configured_llm: bool = False
    lightweight_stats: bool = False

    def check_values(self) -> None:
        if any(not 1 <= value <= 50 for value in self.horizons):
            raise ValueError("Горизонты должны быть от 1 до 50")
        if any(not 0 < value <= 1 for value in self.data_fractions):
            raise ValueError("Доли данных должны быть в интервале (0, 1]")
        for group in (self.horizons, self.data_fractions, self.modes):
            if len(set(group)) != len(group):
                raise ValueError("Параметры эксперимента содержат повторы")
        if self.dataset == "synthetic" and self.seed == 2004:
            raise ValueError("Этот seed зарезервирован для базы знаний")


def benchmark(params: EvalParameters):
    if params.dataset == "synthetic":
        for index in range(params.num_series):
            y, frequency, pattern = synthetic_series(index, params.seed)
            dates = pd.date_range("2000-01-01", periods=len(y), freq=frequency)
            yield f"bench-{index}", y, dates, frequency, pattern
    elif params.dataset == "macro":
        with Path("data/import/us_macrodata.csv").open() as file:
            table = list(csv.DictReader(file))
        dates = pd.to_datetime([f"{row['year']}-{(int(row['quarter']) - 1) * 3 + 1:02}-01" for row in table])
        columns = [column for column in table[0] if column not in {"year", "quarter"}]
        for column in columns[:params.num_series]:
            yield column, np.asarray([float(row[column]) for row in table]), dates, "QS", "macro"
    else:
        table = pd.read_csv("data/import/tatarstan_secondary.csv")
        yield "tatarstan_secondary", table["value"].to_numpy(dtype=float), pd.to_datetime(
            table["date"]).dt.to_period("Q").dt.start_time, "QS", "housing"


def _wilcoxon(difference: np.ndarray, approximate: bool = False) -> float:
    nonzero = difference[np.abs(difference) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.
    if not approximate:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            pass
        else:
            return float(wilcoxon(nonzero).pvalue)
    _, inverse, counts = np.unique(np.abs(nonzero), return_counts=True, return_inverse=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    statistic = abs(float(np.dot(np.sign(nonzero), ranks))) / sqrt(float(ranks @ ranks))
    return erfc(statistic / sqrt(2))


def diebold_mariano(loss_diff: np.ndarray, horizon: int) -> float | None:
    n = len(loss_diff)
    if n < 6:
        return None
    centered = loss_diff - loss_diff.mean()
    lags = min(horizon - 1, n // 3)
    variance = float(centered @ centered) / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag]) / n
    if variance <= 1e-12:
        return None
    return erfc(abs(float(loss_diff.mean())) / sqrt(2 * variance / n))


def _ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = rng.choice(values, size=(1600, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def summarize(rows: list[dict], params: EvalParameters) -> tuple[list[dict], list[dict]]:
    summaries, comparisons = [], []
    rng = np.random.default_rng(110)
    for h, fraction in product(params.horizons, params.data_fractions):
        group = {}
        for mode in params.modes:
            subset = [row for row in rows if row["horizon"] == h and row["fraction"] == fraction
                      and row["mode"] == mode and "metrics" in row]
            if not subset:
                continue
            group[mode] = {row["series_id"]: row for row in subset}
            metrics = {}
            for key in ("mae", "rmse", "mape", "mase", "wape"):
                selected = [row["metrics"][key] for row in subset if row["metrics"][key] is not None]
                metrics[key] = mean(selected) if selected else None
            mase = [row["metrics"]["mase"] for row in subset if row["metrics"]["mase"] is not None]
            summaries.append({
                "mode": mode, "horizon": h, "fraction": fraction, "series": len(subset),
                "metrics": metrics, "mase_ci95": _ci(np.asarray(mase), rng) if mase else None,
                "iterations": mean(row["iterations"] for row in subset),
                "hitl_rate": mean(row["hitl"] for row in subset),
                "tokens": mean(row["tokens"] for row in subset),
                "latency_ms": mean(row["latency_ms"] for row in subset),
            })
        for a, b in combinations(group, 2):
            pairs = [(group[a][key], group[b][key]) for key in sorted(group[a].keys() & group[b].keys())]
            diff = np.asarray([left["metrics"]["mase"] - right["metrics"]["mase"] for left, right in pairs
                               if left["metrics"]["mase"] is not None and right["metrics"]["mase"] is not None])
            dm = [{"series_id": left["series_id"], "pvalue": diebold_mariano(
                np.asarray(left["absolute_errors"]) - np.asarray(right["absolute_errors"]), h)}
                  for left, right in pairs]
            comparisons.append({
                "left": a, "right": b, "horizon": h, "fraction": fraction, "pairs": len(diff),
                "mean_mase_delta": float(diff.mean()) if len(diff) else None,
                "bootstrap_ci95": _ci(diff, rng) if len(diff) > 1 else None,
                "wilcoxon_pvalue": _wilcoxon(diff, params.lightweight_stats) if len(diff) >= 5 else None,
                "diebold_mariano": dm,
            })
    return summaries, comparisons


async def evaluate(params: EvalParameters, settings: Settings) -> dict:
    params.check_values()
    started = perf_counter()
    env = settings if params.configured_llm else settings.model_copy(update={"LLM_PROVIDER": "mock"})
    if params.lightweight_stats:
        env = env.model_copy(update={"STATISTICAL_DIAGNOSTICS": False})
    index = await asyncio.to_thread(load_cases) if any(mode != "baseline" for mode in params.modes) else None
    supervisor = Supervisor(env, HybridRetriever(index, env))
    rows = []
    for key, y, dates, frequency, pattern in benchmark(params):
        for horizon, fraction in product(params.horizons, params.data_fractions):
            end = len(y) - horizon
            n = int(end * fraction)
            for mode in params.modes:
                row = {"series_id": key, "pattern": pattern, "frequency": frequency, "mode": mode,
                       "horizon": horizon, "fraction": fraction, "train_n": n}
                if n < 12:
                    rows.append({**row, "skip": "insufficient_train"})
                    continue
                history = y[end - n:end]
                request = ForecastRequest.model_validate({
                    "data": {"task_type": "timeseries", "frequency": frequency,
                             "nonnegative": bool((history >= 0).all()), "observations": [
                        {"date": day.date(), "value": float(value)} for day, value in
                        zip(dates[end - n:end], history, strict=True)]},
                    "mode": mode, "horizon": horizon,
                })
                outcome = await supervisor.run(key, request)
                result = outcome.result
                actual, predictions = y[end:], np.asarray(result["predictions"])
                rows.append({**row, "model": result["model"]["name"],
                             "metrics": accuracy(actual, predictions, history, PERIODS[frequency]),
                             "absolute_errors": np.abs(actual - predictions).tolist(),
                             "hitl": outcome.needs_review, **result["trajectory"]})
    summary, comparisons = summarize(rows, params)
    return {"parameters": params.model_dump(), "case_base_size": len(index.cases) if index else 0,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "rows": rows, "summary": summary, "comparisons": comparisons}
