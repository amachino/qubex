"""Measurement result model."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field

from qubex.core.model import Model
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)


class MeasurementResult(Model):
    """Canonical serializable result of a measurement run."""

    mode: Literal["single", "avg"]
    data: dict[str, list[np.ndarray]]
    config: dict[str, Any] = Field(default_factory=dict)
    measurement_config: dict[str, Any] = Field(default_factory=dict)
    pulse_metadata: dict[str, Any] = Field(default_factory=dict)
    capture_schedule: CaptureSchedule | None = None

    @property
    def measure_mode(self) -> MeasureMode:
        """Return the mode as `MeasureMode` enum."""
        return MeasureMode(self.mode)

    @classmethod
    def from_multiple(
        cls,
        multiple: MultipleMeasureResult,
    ) -> MeasurementResult:
        """
        Create a `MeasurementResult` from a legacy `MultipleMeasureResult`.

        Parameters
        ----------
        multiple : MultipleMeasureResult
            Legacy multiple-capture result.

        Returns
        -------
        MeasurementResult
            Canonical serializable measurement result.
        """
        data = {
            target: [np.asarray(item.raw) for item in captures]
            for target, captures in multiple.data.items()
        }
        return cls(
            mode=multiple.mode.value,
            data=data,
            config=multiple.config,
        )

    def to_multiple_measure_result(self) -> MultipleMeasureResult:
        """
        Convert to the legacy multi-capture result type.

        Returns
        -------
        MultipleMeasureResult
            Legacy multi-capture result.
        """
        legacy_data = {
            target: [
                MeasureData(
                    target=target,
                    mode=self.measure_mode,
                    raw=np.asarray(raw),
                    classifier=None,
                )
                for raw in captures
            ]
            for target, captures in self.data.items()
        }
        return MultipleMeasureResult(
            mode=self.measure_mode,
            data=legacy_data,
            config=self.config,
        )

    def to_measure_result(
        self,
        *,
        index: int = 0,
    ) -> MeasureResult:
        """
        Convert one capture index to a `MeasureResult`.

        Parameters
        ----------
        index : int, optional
            Capture index in each target's result list.

        Returns
        -------
        MeasureResult
            Per-target result for the selected capture index.

        Raises
        ------
        IndexError
            If `index` is out of range for any target.
        """
        single_data: dict[str, MeasureData] = {}
        for target, captures in self.data.items():
            if not (-len(captures) <= index < len(captures)):
                raise IndexError(
                    f"Capture index {index} is out of range for target {target}."
                )
            single_data[target] = MeasureData(
                target=target,
                mode=self.measure_mode,
                raw=np.asarray(captures[index]),
                classifier=None,
            )

        return MeasureResult(
            mode=self.measure_mode,
            data=single_data,
            config=self.config,
        )
