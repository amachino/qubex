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

    sweep_points: list[SweepPoint] = Field(default_factory=list)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)

    def get(self, index: int) -> MeasurementResult:
        """Return one point result by flat index."""
        return self.results[index]

    def get_sweep_point(self, index: int) -> SweepPoint:
        """Return one input sweep point by flat index."""
        return self.sweep_points[index]


class NDSweepMeasurementResult(DataModel):
    """N-dimensional Cartesian sweep measurement result."""

    sweep_points: dict[SweepKey, list[SweepValue]] = Field(default_factory=dict)
    sweep_axes: SweepAxes = Field(default_factory=tuple)
    shape: tuple[int, ...] = Field(default_factory=tuple)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)

    def get(self, index: int | tuple[int, ...]) -> MeasurementResult:
        """Return one point result by flat index or ndindex."""
        return self.results[self._to_flat_index(index)]

    def get_sweep_point(self, index: int | tuple[int, ...]) -> SweepPoint:
        """Return one resolved sweep point by flat index or ndindex."""
        ndindex = self._to_ndindex(index)
        return {
            axis: self.sweep_points[axis][axis_index]
            for axis, axis_index in zip(self.sweep_axes, ndindex, strict=True)
        }

    def _to_flat_index(self, index: int | tuple[int, ...]) -> int:
        """Normalize flat index or ndindex to flat index."""
        if isinstance(index, int):
            if not (0 <= index < len(self.results)):
                raise IndexError(
                    f"Flat index {index} is out of range for {len(self.results)} results."
                )
            return index

        if len(index) != len(self.shape):
            raise ValueError(
                f"ndindex dimension {len(index)} does not match shape dimension {len(self.shape)}."
            )
        if len(self.shape) == 0:
            return 0

        flat_index = 0
        for axis_index, axis_size in zip(index, self.shape, strict=True):
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

    def _to_ndindex(self, index: int | tuple[int, ...]) -> tuple[int, ...]:
        """Normalize flat index or ndindex to ndindex."""
        if isinstance(index, tuple):
            _ = self._to_flat_index(index)
            return index

        flat_index = self._to_flat_index(index)
        if len(self.shape) == 0:
            return ()

        remaining = flat_index
        ndindex = [0] * len(self.shape)
        for pos in range(len(self.shape) - 1, -1, -1):
            axis_size = self.shape[pos]
            remaining, axis_index = divmod(remaining, axis_size)
            ndindex[pos] = axis_index
        return tuple(ndindex)
