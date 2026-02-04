"""Measurement configuration model."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qubex.core.model import Model
from qubex.measurement.measurement_defaults import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.pulse import RampType


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

    enable_dsp_demodulation: bool = True
    enable_dsp_sum: bool = False
    enable_dsp_classification: bool = False
    line_param0: tuple[float, float, float] | None = None
    line_param1: tuple[float, float, float] | None = None


class MeasurementConfig(Model):
    """Execution configuration for a measurement run."""

    mode: Literal["single", "avg"] = "avg"
    shots: int = DEFAULT_SHOTS
    interval: float = DEFAULT_INTERVAL
    readout: ReadoutConfig = Field(default_factory=ReadoutConfig)
    dsp: DspConfig = Field(default_factory=DspConfig)

    @classmethod
    def create(
        cls,
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MeasurementConfig:
        """Create a run config."""
        return cls(
            mode=mode,
            shots=DEFAULT_SHOTS if shots is None else shots,
            interval=DEFAULT_INTERVAL if interval is None else interval,
            readout=ReadoutConfig(
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
            ),
            dsp=DspConfig(
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
            ),
        )
