"""Shared capture-array shape normalization helpers for measurement adapters."""

from __future__ import annotations

import numpy as np


def normalize_shot_averaged_capture_array(data: object) -> np.ndarray:
    """Return shot-averaged capture payload with singleton axes removed."""
    return np.asarray(data, dtype=np.complex128).squeeze()
