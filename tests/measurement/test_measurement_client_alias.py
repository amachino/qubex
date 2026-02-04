"""Tests for MeasurementClient backward compatibility aliases."""

from __future__ import annotations

from qubex.measurement import Measurement, MeasurementClient
from qubex.measurement.measurement import (
    Measurement as MeasurementFromLegacyModule,
)


def test_measurement_alias_points_to_measurement_client() -> None:
    """Given public aliases, when imported, then they point to MeasurementClient."""
    assert Measurement is MeasurementClient
    assert MeasurementFromLegacyModule is MeasurementClient
