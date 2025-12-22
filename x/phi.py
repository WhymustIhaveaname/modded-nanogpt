#! /usr/bin/env python3

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def phi_fn(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x + b * x**3 + c * x**5


def nest_phi(x: np.ndarray, n: int, a: float, b: float, c: float) -> np.ndarray:
    y = x
    for _ in range(n):
        y = phi_fn(y, a, b, c)
    return y
