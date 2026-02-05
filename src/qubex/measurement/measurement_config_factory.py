"""Factory for building measurement configuration models with contextual defaults."""

from __future__ import annotations

from typing import TypeVar

from qubex.backend import ExperimentSystem
from qubex.pulse import RampType
from qubex.typing import MeasurementMode

from .measurement_defaults import (
    DEFAULT_ENABLE_DSP_CLASSIFICATION,
    DEFAULT_ENABLE_DSP_DEMODULATION,
    DEFAULT_ENABLE_DSP_SUM,
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DRAG_COEFF,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMP_TYPE,
    DEFAULT_READOUT_RAMPTIME,
    DEFAULT_SHOTS,
)
from .models.measurement_config import DspConfig, MeasurementConfig, ReadoutConfig

T = TypeVar("T")


def _or_default(value: T | None, default: T) -> T:
    """Return `default` when value is None; otherwise return value."""
    return default if value is None else value


def _default_readout_amplitudes(
    experiment_system: ExperimentSystem,
) -> dict[str, float]:
    """Return a copy of default readout amplitudes from context."""
    return dict(experiment_system.control_params.readout_amplitude)


class MeasurementConfigFactory:
    """Build `MeasurementConfig` and nested models from partial options."""

    def __init__(
        self,
        *,
        experiment_system: ExperimentSystem,
    ) -> None:
        self._experiment_system: ExperimentSystem = experiment_system

    def create_readout_config(
        self,
        *,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
    ) -> ReadoutConfig:
        """Create `ReadoutConfig` using contextual and configured defaults."""
        return ReadoutConfig(
            readout_amplitudes=_or_default(
                readout_amplitudes,
                _default_readout_amplitudes(self._experiment_system),
            ),
            readout_duration=_or_default(
                readout_duration,
                DEFAULT_READOUT_DURATION,
            ),
            readout_pre_margin=_or_default(
                readout_pre_margin,
                DEFAULT_READOUT_PRE_MARGIN,
            ),
            readout_post_margin=_or_default(
                readout_post_margin,
                DEFAULT_READOUT_POST_MARGIN,
            ),
            readout_ramptime=_or_default(
                readout_ramptime,
                DEFAULT_READOUT_RAMPTIME,
            ),
            readout_drag_coeff=_or_default(
                readout_drag_coeff,
                DEFAULT_READOUT_DRAG_COEFF,
            ),
            readout_ramp_type=_or_default(
                readout_ramp_type,
                DEFAULT_READOUT_RAMP_TYPE,
            ),
        )

    def create_dsp_config(
        self,
        *,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> DspConfig:
        """Create `DspConfig` using configured defaults."""
        return DspConfig(
            enable_dsp_demodulation=_or_default(
                enable_dsp_demodulation,
                DEFAULT_ENABLE_DSP_DEMODULATION,
            ),
            enable_dsp_sum=_or_default(
                enable_dsp_sum,
                DEFAULT_ENABLE_DSP_SUM,
            ),
            enable_dsp_classification=_or_default(
                enable_dsp_classification,
                DEFAULT_ENABLE_DSP_CLASSIFICATION,
            ),
            line_param0=line_param0,
            line_param1=line_param1,
        )

    def create(
        self,
        *,
        mode: MeasurementMode = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MeasurementConfig:
        """Create `MeasurementConfig` from optional runtime overrides."""
        return MeasurementConfig(
            mode=mode,
            shots=_or_default(
                shots,
                DEFAULT_SHOTS,
            ),
            interval=_or_default(
                interval,
                DEFAULT_INTERVAL,
            ),
            readout=self.create_readout_config(
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
            ),
            dsp=self.create_dsp_config(
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
            ),
        )
