"""Константы предметной области: допустимые значения параметров объекта."""

SUPPORTED_REGIONS = ["Kazan"]

HOUSE_TYPES = ["panel", "brick", "monolith", "block", "wood"]

CONDITIONS = ["rough", "needs_repair", "cosmetic", "euro", "designer"]

MIN_HORIZON_MONTHS = 1
MAX_HORIZON_MONTHS = 12

# Доверительный интервал прогноза
INTERVAL_LEVEL = 0.8

# --- Границы для проверок (этап 5) ---
MIN_AREA_M2 = 10.0
MAX_AREA_M2 = 500.0
MIN_BUILD_YEAR = 1800
MAX_FLOORS = 100

# Правдоподобная цена за м² в ₽ — ГРУБЫЕ границы для отсева абсурда.
# Это не рыночные данные: на этапе 6 заменяются статистикой по реальному датасету.
PRICE_M2_MIN = 30_000
PRICE_M2_MAX = 600_000
