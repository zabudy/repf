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

# --- Данные и модель (этап 6) ---
KAZAN_CENTER = (55.7988, 49.1053)  # ориентир (Казанский кремль), широта/долгота
KAZAN_BBOX = {"lat": (55.55, 56.00), "lon": (48.75, 49.45)}  # грубый прямоугольник вокруг города
KAZAN_MAX_DIST_KM = 25.0
MAX_AREA_CLEAN_M2 = 300.0

# Тип дома в датасете: 0 прочее, 1 панель, 2 монолит, 3 кирпич, 4 блок, 5 дерево
BUILDING_TYPE_CODE = {"panel": 1, "monolith": 2, "brick": 3, "block": 4, "wood": 5}
