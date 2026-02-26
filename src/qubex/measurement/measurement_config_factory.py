"""Factory for building measurement configuration models with contextual defaults."""

from __future__ import annotations

from typing import TypeVar

from qubex.system import ExperimentSystem

from .measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOT_AVERAGING,
    DEFAULT_SHOTS,
    DEFAULT_STATE_CLASSIFICATION,
    DEFAULT_TIME_INTEGRATION,
)
from .models.measurement_config import MeasurementConfig

T = TypeVar("T")


def _or_default(value: T | None, default: T) -> T:
    """Return `default` when value is None; otherwise return value."""
    return default if value is None else value


class MeasurementConfigFactory:
    """Build `MeasurementConfig` from partial options."""

    def __init__(
        self,
        *,
        experiment_system: ExperimentSystem,
    ) -> None:
        self._experiment_system: ExperimentSystem = experiment_system

    def create(
        self,
        *,
        n_shots: int | None = None,
        shot_interval_ns: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        """Create `MeasurementConfig` from optional runtime overrides."""
        return MeasurementConfig(
            n_shots=_or_default(
                n_shots,
                DEFAULT_SHOTS,
            ),
            shot_interval_ns=_or_default(
                shot_interval_ns,
                DEFAULT_INTERVAL,
            ),
            shot_averaging=_or_default(
                shot_averaging,
                DEFAULT_SHOT_AVERAGING,
            ),
            time_integration=_or_default(
                time_integration,
                DEFAULT_TIME_INTEGRATION,
            ),
            state_classification=_or_default(
                state_classification,
                DEFAULT_STATE_CLASSIFICATION,
            ),
        )
