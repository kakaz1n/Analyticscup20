"""Small angle / geometry helpers (degrees, ground plane)."""
from __future__ import annotations

import numpy as np


def wrap(a):
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def angdiff(a, b):
    """Signed smallest difference a - b in degrees."""
    return wrap(np.asarray(a) - np.asarray(b))


def bearing(x0, y0, x1, y1):
    """Direction (deg) from (x0, y0) to (x1, y1)."""
    return np.degrees(np.arctan2(np.asarray(y1) - y0, np.asarray(x1) - x0))


def circmean(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    r = np.radians(a)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))
