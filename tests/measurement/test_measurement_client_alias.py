"""Tests for Measurement facade and MeasurementClient compatibility alias."""

from __future__ import annotations

from qubex.measurement import Measurement, MeasurementClient
from qubex.measurement.measurement import (
    Measurement as MeasurementFromPrimaryModule,
)
from qubex.measurement.measurement_client import (
    MeasurementClient as MeasurementClientFromLegacyModule,
)


def test_measurement_is_primary_facade_class() -> None:
    """Given Measurement imports, when loaded, then primary module class is used."""
    assert Measurement is MeasurementFromPrimaryModule


def test_measurement_client_alias_points_to_measurement() -> None:
    """Given legacy MeasurementClient imports, they remain compatible aliases."""
    assert MeasurementClient is MeasurementClientFromLegacyModule
    assert issubclass(MeasurementClient, Measurement)
