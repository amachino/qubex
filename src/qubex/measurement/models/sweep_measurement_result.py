from __future__ import annotations

from numpy.typing import NDArray

from qubex.core.model import Model


class SweepMeasurementResult(Model):
    metadata: dict
    data: NDArray
    data_shape: list[int]
    sweep_key_list: list[str]
    data_key_list: list[str]
