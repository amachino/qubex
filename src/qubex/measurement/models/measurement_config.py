"""Measurement configuration model."""

from __future__ import annotations

from qubex.core import Model
from qubex.typing import MeasurementMode


class MeasurementConfig(Model):
    """Measurement configuration model."""

    mode: MeasurementMode
    shots: int
    interval: float
    enable_dsp_demodulation: bool
    enable_dsp_sum: bool
    enable_dsp_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]
