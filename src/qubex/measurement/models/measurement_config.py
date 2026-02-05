"""Measurement configuration model."""

from __future__ import annotations

from qubex.core.model import Model
from qubex.typing import MeasurementMode


class DspConfig(Model):
    """DSP configuration model."""

    enable_dsp_demodulation: bool
    enable_dsp_sum: bool
    enable_dsp_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]


class FrequencyConfig(Model):
    """Frequency configuration model."""

    frequencies: dict[str, float]


class MeasurementConfig(Model):
    """Measurement configuration model."""

    mode: MeasurementMode
    shots: int
    interval: float
    dsp: DspConfig
    frequency: FrequencyConfig
