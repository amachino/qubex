"""Measurement result model."""

from __future__ import annotations

from typing import Any, cast

from qubex.core.model import Model
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)


class MeasurementResult(Model):
    """High-level result wrapper for measurement runs."""

    multiple: Any

    @property
    def mode(self) -> MeasureMode:
        """Return the measurement mode."""
        return self.to_multiple_measure_result().mode

    @property
    def config(self) -> dict:
        """Return backend-related run configuration."""
        return self.to_multiple_measure_result().config

    def to_multiple_measure_result(self) -> MultipleMeasureResult:
        """
        Return the wrapped multi-capture measurement result.

        Returns
        -------
        MultipleMeasureResult
            Wrapped result for all captures.
        """
        if not isinstance(self.multiple, MultipleMeasureResult):
            raise TypeError(
                "MeasurementResult.multiple must be a MultipleMeasureResult instance."
            )
        return cast(MultipleMeasureResult, self.multiple)

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
        TypeError
            If the wrapped result is not a `MultipleMeasureResult`.
        """
        multiple = self.to_multiple_measure_result()
        single_data: dict[str, MeasureData] = {}
        for target, data_list in multiple.data.items():
            if not (-len(data_list) <= index < len(data_list)):
                raise IndexError(
                    f"Capture index {index} is out of range for target {target}."
                )
            single_data[target] = data_list[index]

        return MeasureResult(
            mode=multiple.mode,
            data=single_data,
            config=multiple.config,
        )
