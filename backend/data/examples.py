"""Deterministic synthetic series for the case-base and an independent demo benchmark."""
import numpy as np

PATTERNS = ("seasonal", "trend", "autoregressive", "mixed", "level_shift")
PERIODS = {"D": 7, "W": 52, "MS": 12, "QS": 4}


def synthetic_series(index: int, seed: int, length: int = 260) -> tuple[np.ndarray, str, str]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    frequency = list(PERIODS)[index % 4]
    period = PERIODS[frequency]
    pattern = PATTERNS[index % len(PATTERNS)]
    t = np.arange(length)
    base = float(rng.uniform(40, 300))
    noise = rng.normal(0, base * rng.uniform(.01, .12), length)
    if pattern == "autoregressive":
        phi = rng.uniform(.3, .96)
        for i in range(1, length):
            noise[i] += phi * noise[i - 1]
    trend = float(rng.uniform(-.002, .010)) if pattern in {"trend", "mixed"} else 0
    amplitude = float(rng.uniform(.08, .25)) if pattern in {"seasonal", "mixed"} else 0
    level_change = (t >= rng.integers(length // 3, length * 3 // 4)) * base * rng.uniform(-.25, .35)
    values = base * (1 + trend * t + amplitude * np.sin(2 * np.pi * t / period + rng.uniform(0, 7)))
    return np.maximum(.1, values + noise + (level_change if pattern == "level_shift" else 0)), frequency, pattern
