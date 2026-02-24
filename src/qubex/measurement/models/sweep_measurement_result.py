"""Models for sweep measurement execution results."""

from __future__ import annotations

from pydantic import Field

from qubex.core import DataModel, Model, Value

from .measurement_result import MeasurementResult


class SweepPoint(Model):
    """One sweep point."""

    parameters: dict[str, Value | int | float | str] = Field(default_factory=dict)


class SweepPointResult(DataModel):
    """Result for one sweep point."""

    index: int
    point: SweepPoint
    result: MeasurementResult


class SweepMeasurementResult(DataModel):
    """Sweep measurement result."""

    results: list[SweepPointResult] = Field(default_factory=list)
