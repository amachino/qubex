"""Measurement configuration model."""

from __future__ import annotations

from qubex.core.model import Model
from qubex.pulse import RampType
from qubex.typing import MeasurementMode


class DspConfig(Model):
    """DSP configuration model."""

    enable_dsp_demodulation: bool
    enable_dsp_sum: bool
    enable_dsp_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]


class ReadoutConfig(Model):
    """Readout pulse configuration model."""

    readout_amplitudes: dict[str, float]
    readout_duration: float
    readout_pre_margin: float
    readout_post_margin: float
    readout_ramptime: float
    readout_drag_coeff: float
    readout_ramp_type: RampType


class MeasurementConfig(Model):
    """Measurement configuration model."""

    mode: MeasurementMode
    shots: int
    interval: float
    dsp: DspConfig
    readout: ReadoutConfig
