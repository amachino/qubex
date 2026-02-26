"""Models for sweep measurement execution results."""

from __future__ import annotations

from pydantic import Field

from qubex.core import DataModel, Value

from .measurement_config import MeasurementConfig
from .measurement_result import MeasurementResult

SweepKey = str
SweepValue = Value | int | float | str
SweepPoint = dict[SweepKey, SweepValue]
SweepAxes = tuple[SweepKey, ...]


class SweepMeasurementResult(DataModel):
    """Sweep measurement result."""

    sweep_values: list[SweepValue] = Field(default_factory=list)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)


class NDSweepMeasurementResult(DataModel):
    """N-dimensional Cartesian sweep measurement result."""

    sweep_points: dict[SweepKey, list[SweepValue]] = Field(default_factory=dict)
    sweep_axes: SweepAxes = Field(default_factory=tuple)
    shape: tuple[int, ...] = Field(default_factory=tuple)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)

    def get(self, ndindex: tuple[int, ...]) -> MeasurementResult:
        """Return one point result by ndindex."""
        return self.results[self._to_flat_index(ndindex)]

    def get_sweep_point(self, ndindex: tuple[int, ...]) -> SweepPoint:
        """Return one resolved sweep point by ndindex."""
        _ = self._to_flat_index(ndindex)
        return {
            axis: self.sweep_points[axis][axis_index]
            for axis, axis_index in zip(self.sweep_axes, ndindex, strict=True)
        }

    def _to_flat_index(self, ndindex: tuple[int, ...]) -> int:
        """Normalize ndindex to flat index."""
        if len(ndindex) != len(self.shape):
            raise ValueError(
                f"ndindex dimension {len(ndindex)} does not match shape dimension {len(self.shape)}."
            )
        if len(self.shape) == 0:
            return 0

        flat_index = 0
        for axis_index, axis_size in zip(ndindex, self.shape, strict=True):
            if not (0 <= axis_index < axis_size):
                raise IndexError(
                    f"ndindex element {axis_index} is out of bounds for axis size {axis_size}."
                )
            flat_index = flat_index * axis_size + axis_index

        if not (0 <= flat_index < len(self.results)):
            raise IndexError(
                f"Flattened index {flat_index} is out of range for {len(self.results)} results."
            )
        return flat_index
