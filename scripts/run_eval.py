"""Compare forecasting modes on temporal holdouts and export JSON, CSV and SVG."""
import argparse
import asyncio
import csv
import logging
from pathlib import Path
from xml.sax.saxutils import escape

from backend.core.config import get_settings
from backend.evaluation.runner import EvalParameters, MODES, evaluate
from scripts.generate_knowledge_base import atomic_json


def chart(results: dict, path: Path) -> None:
    rows = [row for row in results["summary"] if row["metrics"]["mase"] is not None]
    if not rows:
        return
    width, height, x0, y0 = 760, 390, 78, 315
    scale = max(row["metrics"]["mase"] for row in rows) * 1.16 or 1
    horizons = sorted({row["horizon"] for row in rows})
    colors = dict(zip(MODES, ["#374151", "#447aee", "#137e65"], strict=True))
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 760 390">',
              '<rect width="760" height="390" fill="#f7f9fc"/>',
              '<g font-family="sans-serif" fill="#243047">',
              '<text x="30" y="36" font-size="20">Mean MASE by training fraction</text>']
    for i in range(6):
        y = y0 - i * 45
        output.extend([f'<path d="M{x0} {y} H720" stroke="#dce3ee"/>',
                       f'<text x="36" y="{y + 4}" font-size="12">{scale * i / 5:.2f}</text>'])
    fractions = sorted({row["fraction"] for row in rows})
    panel = 625 / len(horizons)
    for k, horizon in enumerate(horizons):
        left = x0 + k * panel + 13
        span = panel - 28
        positions = {f: left + span * i / max(1, len(fractions) - 1) for i, f in enumerate(fractions)}
        output.append(f'<text x="{left}" y="75" font-size="13">Horizon {horizon}</text>')
        for f, x in positions.items():
            output.append(f'<text x="{x - 10}" y="335" font-size="11">{f:.0%}</text>')
        for mode, color in colors.items():
            points = sorted((row["fraction"], row["metrics"]["mase"]) for row in rows
                            if row["horizon"] == horizon and row["mode"] == mode)
            coords = [(positions[f], y0 - value / scale * 225) for f, value in points]
            output.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="' +
                          " ".join(f"{x},{y}" for x, y in coords) + '"/>')
            output.extend(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>' for x, y in coords)
    for x, (mode, color) in zip([78, 220, 360], colors.items(), strict=True):
        output.append(f'<text x="{x}" y="367" fill="{color}" font-size="13">{escape(mode)}</text>')
    output.append("</g></svg>")
    path.write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-series", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--data-fractions", type=float, nargs="+", default=[.1, .2, .5, 1.])
    parser.add_argument("--mode", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--dataset", choices=["synthetic", "macro", "tatarstan"], default="synthetic")
    parser.add_argument("--seed", type=int, default=5003)
    parser.add_argument("--configured-llm", action="store_true")
    parser.add_argument("--lightweight-stats", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/evaluation.json"))
    args = parser.parse_args()
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(filename="logs/eval.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    params = EvalParameters(num_series=args.num_series, horizons=args.horizons,
                            data_fractions=args.data_fractions, modes=args.mode, dataset=args.dataset,
                            seed=args.seed, configured_llm=args.configured_llm,
                            lightweight_stats=args.lightweight_stats)
    logging.info("Starting %s", params.model_dump_json())
    report = asyncio.run(evaluate(params, get_settings()))
    atomic_json(args.output, report)
    summary_fields = ["mode", "horizon", "fraction", "series", "mae", "rmse", "mape", "mase", "wape",
                      "iterations", "hitl_rate", "tokens", "latency_ms"]
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, **row["metrics"]} for row in report["summary"])
    chart(report, args.output.with_suffix(".svg"))
    skipped = sum("skip" in row for row in report["rows"])
    logging.info("Finished rows=%d skipped=%d elapsed=%s", len(report["rows"]), skipped, report["elapsed_seconds"])
    print(f"{len(report['rows']) - skipped} scored runs, {skipped} skipped; {args.output}")


if __name__ == "__main__":
    main()
