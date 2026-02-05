"""Factory for building measurement configuration models with contextual defaults."""

from __future__ import annotations

from typing import TypeVar

from qubex.backend import ExperimentSystem
from qubex.typing import MeasurementMode

from .measurement_defaults import (
    DEFAULT_ENABLE_DSP_CLASSIFICATION,
    DEFAULT_ENABLE_DSP_DEMODULATION,
    DEFAULT_ENABLE_DSP_SUM,
    DEFAULT_INTERVAL,
    DEFAULT_LINE_PARAM0,
    DEFAULT_LINE_PARAM1,
    DEFAULT_SHOTS,
)
from .models.measurement_config import (
    DspConfig,
    FrequencyConfig,
    MeasurementConfig,
)

T = TypeVar("T")


def _or_default(value: T | None, default: T) -> T:
    """Return `default` when value is None; otherwise return value."""
    return default if value is None else value


class MeasurementConfigFactory:
    """Build `MeasurementConfig` and nested models from partial options."""

    def __init__(
        self,
        *,
        experiment_system: ExperimentSystem,
    ) -> None:
        self._experiment_system: ExperimentSystem = experiment_system

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
            line_param0=_or_default(
                line_param0,
                DEFAULT_LINE_PARAM0,
            ),
            line_param1=_or_default(
                line_param1,
                DEFAULT_LINE_PARAM1,
            ),
        )

    def create_frequency_config(
        self,
        *,
        frequencies: dict[str, float] | None = None,
    ) -> FrequencyConfig:
        """Create `FrequencyConfig` using configured defaults."""
        return FrequencyConfig(
            frequencies=_or_default(
                frequencies,
                {},
            ),
        )

    def create(
        self,
        *,
        mode: MeasurementMode = "avg",
        shots: int | None = None,
        interval: float | None = None,
        frequencies: dict[str, float] | None = None,
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
            frequency=self.create_frequency_config(
                frequencies=frequencies,
            ),
            dsp=self.create_dsp_config(
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
            ),
        )
