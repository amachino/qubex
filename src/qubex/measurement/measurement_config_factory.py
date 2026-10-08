"""Factory for building measurement configuration models with contextual defaults."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from qubex.system import ExperimentSystem

from .measurement_defaults import (
    DEFAULT_SHOT_AVERAGING,
    DEFAULT_STATE_CLASSIFICATION,
    DEFAULT_TIME_INTEGRATION,
    resolve_measurement_defaults,
)
from .models.measurement_config import MeasurementConfig, ReturnItem

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
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        return_items: Sequence[ReturnItem] | None = None,
    ) -> MeasurementConfig:
        """Create `MeasurementConfig` from optional runtime overrides."""
        resolved_return_items: tuple[ReturnItem, ...]
        if return_items is None:
            resolved_return_items = ()
        else:
            resolved_return_items = tuple(return_items)
        measurement_defaults = resolve_measurement_defaults(
            getattr(self._experiment_system, "measurement_defaults", None)
        )

        return MeasurementConfig(
            n_shots=_or_default(
                n_shots,
                measurement_defaults.execution.n_shots,
            ),
            shot_interval=_or_default(
                shot_interval,
                measurement_defaults.execution.shot_interval_ns,
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
            return_items=resolved_return_items,
        )
