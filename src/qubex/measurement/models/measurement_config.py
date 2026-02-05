"""Measurement configuration model."""

from __future__ import annotations

from qubex.core.model import Model
from qubex.pulse import RampType
from qubex.typing import MeasurementMode


class ReadoutConfig(Model):
    """Readout pulse configuration."""

    readout_amplitudes: dict[str, float] | None = None
    readout_duration: float | None = None
    readout_pre_margin: float | None = None
    readout_post_margin: float | None = None
    readout_ramptime: float | None = None
    readout_drag_coeff: float | None = None
    readout_ramp_type: RampType | None = None


class DspConfig(Model):
    """DSP configuration."""

    enable_dsp_demodulation: bool
    enable_dsp_sum: bool
    enable_dsp_classification: bool
    line_param0: tuple[float, float, float] | None = None
    line_param1: tuple[float, float, float] | None = None


class MeasurementConfig(Model):
    """Execution configuration for a measurement run."""

    mode: MeasurementMode
    shots: int
    interval: float
    readout: ReadoutConfig
    dsp: DspConfig
