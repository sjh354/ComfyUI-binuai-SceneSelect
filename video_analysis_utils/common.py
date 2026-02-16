from __future__ import annotations

from typing import Dict, List

import numpy as np


def safe_float(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(x)


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": safe_float(np.mean(arr)),
        "median": safe_float(np.median(arr)),
        "std": safe_float(np.std(arr)),
        "min": safe_float(np.min(arr)),
        "max": safe_float(np.max(arr)),
    }
