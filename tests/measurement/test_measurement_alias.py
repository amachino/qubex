"""Tests for Measurement facade imports."""

from __future__ import annotations

from qubex.measurement import Measurement
from qubex.measurement.measurement import (
    Measurement as MeasurementFromPrimaryModule,
)


def test_measurement_is_primary_facade_class() -> None:
    """Given Measurement imports, when loaded, then primary module class is used."""
    assert Measurement is MeasurementFromPrimaryModule
