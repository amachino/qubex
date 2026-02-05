"""Factory for building measurement configuration models with contextual defaults."""

from __future__ import annotations

from typing import cast

from qubex.backend import ExperimentSystem
from qubex.pulse import RampType
from qubex.typing import MeasurementMode

from .measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMPTIME,
    DEFAULT_SHOTS,
)
from .models.measurement_config import DspConfig, MeasurementConfig, ReadoutConfig


class MeasurementConfigFactory:
    """Build `MeasurementConfig` and nested models from partial options."""

    def __init__(
        self,
        *,
        experiment_system: ExperimentSystem,
        measurement_defaults: dict[str, object] | None = None,
    ) -> None:
        self._experiment_system: ExperimentSystem = experiment_system
        defaults: dict[str, object] = (
            {} if measurement_defaults is None else dict(measurement_defaults)
        )
        self._shots_default: int = cast(int, defaults.get("shots", DEFAULT_SHOTS))
        self._interval_default: float = cast(
            float, defaults.get("interval", DEFAULT_INTERVAL)
        )
        self._readout_duration_default: float = cast(
            float, defaults.get("readout_duration", DEFAULT_READOUT_DURATION)
        )
        self._readout_pre_margin_default: float = cast(
            float, defaults.get("readout_pre_margin", DEFAULT_READOUT_PRE_MARGIN)
        )
        self._readout_post_margin_default: float = cast(
            float, defaults.get("readout_post_margin", DEFAULT_READOUT_POST_MARGIN)
        )
        self._readout_ramptime_default: float = cast(
            float, defaults.get("readout_ramptime", DEFAULT_READOUT_RAMPTIME)
        )
        self._readout_drag_coeff_default: float = cast(
            float, defaults.get("readout_drag_coeff", 0.0)
        )
        self._readout_ramp_type_default: RampType = cast(
            RampType, defaults.get("readout_ramp_type", "RaisedCosine")
        )
        self._enable_dsp_demodulation_default: bool = cast(
            bool, defaults.get("enable_dsp_demodulation", True)
        )
        self._enable_dsp_sum_default: bool = cast(
            bool, defaults.get("enable_dsp_sum", False)
        )
        self._enable_dsp_classification_default: bool = cast(
            bool, defaults.get("enable_dsp_classification", False)
        )

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
            readout_amplitudes=(
                self._default_readout_amplitudes()
                if readout_amplitudes is None
                else readout_amplitudes
            ),
            readout_duration=(
                self._readout_duration_default
                if readout_duration is None
                else readout_duration
            ),
            readout_pre_margin=(
                self._readout_pre_margin_default
                if readout_pre_margin is None
                else readout_pre_margin
            ),
            readout_post_margin=(
                self._readout_post_margin_default
                if readout_post_margin is None
                else readout_post_margin
            ),
            readout_ramptime=(
                self._readout_ramptime_default
                if readout_ramptime is None
                else readout_ramptime
            ),
            readout_drag_coeff=(
                self._readout_drag_coeff_default
                if readout_drag_coeff is None
                else readout_drag_coeff
            ),
            readout_ramp_type=(
                self._readout_ramp_type_default
                if readout_ramp_type is None
                else readout_ramp_type
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
            enable_dsp_demodulation=(
                self._enable_dsp_demodulation_default
                if enable_dsp_demodulation is None
                else enable_dsp_demodulation
            ),
            enable_dsp_sum=(
                self._enable_dsp_sum_default
                if enable_dsp_sum is None
                else enable_dsp_sum
            ),
            enable_dsp_classification=(
                self._enable_dsp_classification_default
                if enable_dsp_classification is None
                else enable_dsp_classification
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
            shots=self._shots_default if shots is None else shots,
            interval=self._interval_default if interval is None else interval,
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

    def _default_readout_amplitudes(self) -> dict[str, float]:
        """Return a copy of default readout amplitudes from context."""
        return dict(self._experiment_system.control_params.readout_amplitude)
