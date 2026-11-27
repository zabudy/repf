"""Обучить и оценить модель: python -m scripts.train [--synthetic]"""
import argparse

import pandas as pd

from backend.core.config import get_settings
from backend.data.loader import clean, month_label
from backend.data.synthetic import make_synthetic
from backend.models.hedonic import save_bundle, train_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/kazan.csv")
    ap.add_argument("--synthetic", action="store_true", help="ДЕМО-данные вместо реальных")
    ap.add_argument("--out", default=get_settings().MODEL_PATH)
    a = ap.parse_args()

    if a.synthetic:
        print("!!! ДЕМО-РЕЖИМ: данные синтетические, результат не отражает реальный рынок !!!")
        df, _ = clean(make_synthetic(30_000))
    else:
        df = pd.read_csv(a.data)
        df["date"] = pd.to_datetime(df["date"])
        print(f"Загружено {len(df):,} строк из {a.data}")

    print(f"Период данных: {month_label(df['month'].min())} … {month_label(df['month'].max())}")
    print("Обучение (может занять несколько минут)...")
    bundle, m = train_pipeline(df)

    print("\n=== Качество на самых новых месяцах (honest time split) ===")
    print(f"Строк: train {m['rows_train']:,} / calib {m['rows_calib']:,} / test {m['rows_test']:,}")
    print(f"MAPE модели:                {m['mape_model']:.1%}")
    print(f"MAPE базового (медиана м²): {m['mape_baseline_median']:.1%}")
    print(f"MAE:  {m['mae_rub']:,.0f} ₽    RMSE: {m['rmse_rub']:,.0f} ₽")
    print(f"Покрытие 80 % интервала:    {m['interval_coverage_80']:.1%}")

    save_bundle(bundle, a.out)
    print(f"\nМодель сохранена: {a.out}")
