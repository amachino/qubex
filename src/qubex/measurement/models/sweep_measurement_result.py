"""Models for sweep measurement execution results."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from qubex.core import DataModel, Value

from .measurement_config import MeasurementConfig
from .measurement_result import MeasurementResult

SweepKey = str
SweepValue = Value | int | float | str
SweepPoint = dict[SweepKey, SweepValue]
SweepAxes = tuple[SweepKey, ...]


def _build_sweep_data(
    *,
    results: list[MeasurementResult],
    sweep_shape: tuple[int, ...],
) -> dict[str, list[NDArray[Any]]]:
    """Aggregate capture arrays as `target -> [capture_index arrays]` with sweep axes."""
    expected_points = int(np.prod(sweep_shape, dtype=int))
    if len(results) != expected_points:
        raise ValueError(
            "results length does not match sweep shape: "
            f"len(results)={len(results)}, sweep_shape={sweep_shape}."
        )
    if len(results) == 0:
        return {}

    expected_capture_counts: dict[str, int] = {}
    per_target_capture_series: dict[str, list[list[NDArray[Any]]]] = {}

    for point_index, result in enumerate(results):
        result_targets = set(result.data)
        if point_index > 0:
            missing_targets = sorted(set(expected_capture_counts) - result_targets)
            if missing_targets:
                joined = ", ".join(missing_targets)
                raise ValueError(
                    f"Missing targets at sweep point index {point_index}: {joined}."
                )
        for target, captures in result.data.items():
            capture_count = len(captures)
            expected = expected_capture_counts.get(target)
            if expected is None:
                expected_capture_counts[target] = capture_count
                per_target_capture_series[target] = [[] for _ in range(capture_count)]
            elif expected != capture_count:
                raise ValueError(
                    f"Capture count mismatch for target {target} at sweep point "
                    f"index {point_index}: expected {expected}, got {capture_count}."
                )
            for capture_index, capture in enumerate(captures):
                per_target_capture_series[target][capture_index].append(
                    np.asarray(capture.data)
                )

    sweep_data: dict[str, list[NDArray[Any]]] = {}
    for target, capture_series_list in per_target_capture_series.items():
        sweep_data[target] = []
        for capture_series in capture_series_list:
            stacked = np.stack(capture_series, axis=0)
            reshaped = stacked.reshape((*sweep_shape, *stacked.shape[1:]))
            sweep_data[target].append(reshaped)
    return sweep_data


class SweepMeasurementResult(DataModel):
    """Sweep measurement result."""

    sweep_values: list[SweepValue] = Field(default_factory=list)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)

    @property
    def data(self) -> dict[str, list[NDArray[Any]]]:
        """Return target-keyed capture arrays in sweep order."""
        return _build_sweep_data(
            results=self.results,
            sweep_shape=(len(self.sweep_values),),
        )


class NDSweepMeasurementResult(DataModel):
    """N-dimensional Cartesian sweep measurement result."""

    sweep_points: dict[SweepKey, list[SweepValue]] = Field(default_factory=dict)
    sweep_axes: SweepAxes = Field(default_factory=tuple)
    shape: tuple[int, ...] = Field(default_factory=tuple)
    config: MeasurementConfig
    results: list[MeasurementResult] = Field(default_factory=list)

    @property
    def data(self) -> dict[str, list[NDArray[Any]]]:
        """Return target-keyed capture arrays in flattened C-order."""
        return _build_sweep_data(
            results=self.results,
            sweep_shape=self.shape,
        )

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
