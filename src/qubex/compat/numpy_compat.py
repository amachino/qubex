"""Compatibility helpers for NumPy API changes."""

import numpy as np

try:
    trapezoid = np.trapezoid  # pyright: ignore[reportAttributeAccessIssue]
except AttributeError:
    trapezoid = np.trapz  # pyright: ignore[reportAttributeAccessIssue]
