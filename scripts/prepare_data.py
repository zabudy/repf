"""Оставить только Казань и очистить: python -m scripts.prepare_data"""
import argparse
import logging
from pathlib import Path

from backend.data.loader import clean, month_label, read_kazan_raw

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/russia_real_estate.csv")
    ap.add_argument("--output", default="data/processed/kazan.csv")
    a = ap.parse_args()

    raw = read_kazan_raw(a.input)
    df, stats = clean(raw)
    print("\nЭтапы очистки (строк):")
    for k, v in stats.items():
        print(f"  {k:22} {v:>10,}")
    if df.empty:
        raise SystemExit("После очистки ничего не осталось — проверьте файл.")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"\nПериод: {month_label(df['month'].min())} … {month_label(df['month'].max())}")
    print(f"Медиана цены за м²: {df['price_m2'].median():,.0f} ₽")
    print(f"Сохранено: {a.output}")
