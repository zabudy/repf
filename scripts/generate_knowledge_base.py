"""Train a small grid on historical prefixes and export reusable cases."""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.agents.data_analyst import analyse
from backend.core.forecast import Apartment, TabularInput, ToolChoice
from backend.core.llm.tools import available, series_prediction, validate
from backend.core.metrics import accuracy
from backend.data.examples import PERIODS, synthetic_series
from backend.retrieval.hybrid import CASE_PATH, describe
from backend.tabular.housing import backtest, demo_sales
from backend.timeseries.preprocessing import acf, feature_vector

GLOSSARY_PATH = CASE_PATH.with_name("glossary.json")
GRID = (
    ("fit_holt", {}), ("fit_holt", {"alpha": .65, "damping": .98}),
    ("fit_harmonic", {"harmonics": 2}), ("fit_harmonic", {"harmonics": 5}),
    ("fit_autoregression", {"differences": 0, "lags": 8}),
    ("fit_autoregression", {"differences": 1, "lags": 16}),
)


def series_case(case_id: str, values: np.ndarray, frequency: str, pattern: str,
                source: str) -> dict:
    period = PERIODS[frequency]
    horizon = min(20, max(3, len(values) // 8))
    y, test = values[:-horizon], values[-horizon:]
    choices = []
    for name, args in GRID:
        tool = validate(ToolChoice(name=name, arguments=args), "timeseries")
        forecast = series_prediction(tool, y, horizon, period)
        choices.append((accuracy(test, forecast, y, period), tool))
    metrics, best = min(choices, key=lambda choice: choice[0]["mae"])
    meta = {
        "task_type": "timeseries", "frequency": frequency, "period": period,
        "trend": float((y[-6:].mean() - y[:6].mean()) / max(y.std(), 1e-8)),
        "seasonality": acf(y, period), "autocorrelation": acf(y, 1), "source": source,
    }
    return {"id": case_id, "metadata": meta, "features": feature_vector(y).tolist(),
            "description": f"{pattern}. {describe(meta)}. {best.name} won on {horizon} held-out points.",
            "best_model": best.name, "arguments": best.arguments, "metrics": metrics}


def tabular_cases(count: int) -> list[dict]:
    result = []
    for batch in range((count + 11) // 12):
        sales = demo_sales(seed=1000 + batch, months=36, per_month=10)
        for sample in range(min(12, count - len(result))):
            apartment = Apartment.model_validate(sales[sample * 9].model_dump(
                exclude={"date", "price_rub"}))
            market, meta, features = analyse(TabularInput(apartment=apartment, market=sales))
            if sample == 0:
                options = []
                for name in available("tabular"):
                    tool = validate(ToolChoice(name=name), "tabular")
                    actual, predicted, history = backtest(market, name, tool.arguments)
                    options.append((accuracy(actual, predicted, history, 1), tool))
                metrics, best = min(options, key=lambda row: row[0]["mae"])
            meta["source"] = "synthetic_housing"
            result.append({"id": f"housing-{batch:02}-{sample:02}", "metadata": meta,
                           "features": features.tolist(), "best_model": best.name,
                           "arguments": best.arguments, "metrics": metrics,
                           "description": f"{describe(meta)}. {best.name} won on the later months."})
    return result


CHECKS = {
    "MAE": "Средняя абсолютная ошибка в исходных единицах. Меньше — лучше.",
    "RMSE": "Корень из средней квадратичной ошибки. Сильнее реагирует на крупные промахи.",
    "MAPE": "Средний модуль относительной ошибки в процентах. Нулевые цели исключаются.",
    "MASE": "MAE, делённая на среднюю абсолютную наивную ошибку на обучающем участке.",
    "WAPE": "Сумма абсолютных ошибок, делённая на сумму абсолютных фактов.",
    "Bias": "Среднее знаковое отклонение прогноза от факта. Показывает направление смещения.",
    "Median AE": "Медиана абсолютных ошибок, устойчивая к редким выбросам.",
    "P90 AE": "90-й процентиль абсолютных ошибок, характеризующий крупные промахи.",
    "Coverage": "Доля фактов внутри предсказательного интервала.",
    "Interval width": "Разность верхней и нижней границ предсказательного интервала.",
    "ACF": "Корреляция ряда с собственными лагами.",
    "PACF": "Частичная автокорреляция после учёта более коротких лагов.",
    "Ljung–Box": "Тест совместной значимости автокорреляций на группе лагов.",
    "Shapiro–Wilk": "Тест согласия распределения с нормальным.",
    "ADF": "Тест нулевой гипотезы о наличии единичного корня.",
    "STL": "Разложение ряда на тренд, сезонность и остаток через локальное сглаживание.",
    "Variance": "Средний квадрат отклонения от среднего, мера разброса.",
    "IQR": "Разность третьего и первого квартилей, устойчивая мера разброса.",
    "Drift": "Изменение распределения входов или цели между периодами.",
    "Missing rate": "Доля пропущенных наблюдений или полей.",
    "Sample size": "Число наблюдений, вошедших в расчёт.",
    "Bootstrap CI": "Интервал, полученный повторными выборками с возвращением.",
    "Wilcoxon": "Парный тест на основе рангов ненулевых разностей.",
    "Diebold–Mariano": "Тест равной ожидаемой функции потерь двух прогнозов.",
    "RRF": "Суммирование обратных рангов одного кейса в выдачах ретриверов.",
}
SCOPES = {
    "обучающего окна": "Используйте только историю, доступную до момента прогноза.",
    "валидационного окна": "Используйте для выбора конфигурации, оставив финальный тест закрытым.",
    "финального теста": "Выполните расчёт после фиксации моделей и параметров.",
    "первого шага": "Оставьте только ближайшую точку для каждого момента запуска.",
    "длинного горизонта": "Сравнивайте модели на одинаковом числе будущих шагов.",
    "дневного ряда": "Сохраняйте дневную гранулярность и календарный порядок.",
    "недельного ряда": "Фиксируйте одинаковое правило конца недели.",
    "месячного ряда": "Сравнивайте одинаковые календарные месяцы.",
    "квартального ряда": "Отмечайте, к какому кварталу относится каждое значение.",
    "сезонного ряда": "Фиксируйте период сезонности и количество доступных циклов.",
    "ряда со структурным сдвигом": "Отдельно проверьте участки до и после сдвига.",
    "ряда с пропусками": "Отмечайте импутированные периоды и метод заполнения.",
    "нуля в наблюдениях": "Проверьте применимость относительных формул.",
    "подвыборки 10%": "Сохраняйте последние 10% обучающей истории и тот же тест.",
    "подвыборки 20%": "Сохраняйте последние 20% обучающей истории и тот же тест.",
    "подвыборки 50%": "Сохраняйте последние 50% обучающей истории и тот же тест.",
    "нового района": "Отмечайте недостаток наблюдений по новому сегменту.",
    "сегмента квартир": "Фиксируйте критерии сегмента и размер подвыборки.",
    "набора похожих кейсов": "Исключайте целевой ряд из базы схожих примеров.",
    "ручной проверки": "Сохраняйте значение до проверки и решение эксперта.",
}


def glossary() -> list[dict]:
    # The 500-entry matrix is a reference of terms in concrete evaluation settings.
    return [{"term": f"{name} — {scope}", "definition": f"{definition} {usage}"}
            for name, definition in CHECKS.items() for scope, usage in SCOPES.items()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=1024)
    parser.add_argument("--housing", type=int, default=144)
    parser.add_argument("--output", type=Path, default=CASE_PATH)
    args = parser.parse_args()
    if args.series < 1 or args.housing < 0:
        parser.error("Require --series >= 1 and --housing >= 0")
    cases = []
    for i in range(args.series):
        y, freq, pattern = synthetic_series(i, seed=2004)
        cases.append(series_case(f"synthetic-{i:04}", y, freq, pattern, "synthetic"))
    cases.extend(tabular_cases(args.housing))
    terms = glossary()
    atomic_json(args.output, cases)
    atomic_json(args.output.with_name(GLOSSARY_PATH.name), terms)
    print(f"{len(cases)} cases and {len(terms)} glossary entries exported")


if __name__ == "__main__":
    main()
