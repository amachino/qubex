"""Shared typing aliases for qxpulse."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

IQArray: TypeAlias = (
    list[complex] | list[float] | NDArray[np.complexfloating] | NDArray[np.floating]
)
