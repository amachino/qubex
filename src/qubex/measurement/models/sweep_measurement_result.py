"""Result model for sweep measurements."""

from __future__ import annotations

from numpy.typing import NDArray
from qxcore.model import Model


class SweepMeasurementResult(Model):
    """Container for sweep measurement result data."""

    metadata: dict
    data: NDArray
    data_shape: list[int]
    sweep_key_list: list[str]
    data_key_list: list[str]
