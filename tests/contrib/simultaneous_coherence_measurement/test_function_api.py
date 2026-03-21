"""Tests for functional APIs in `qubex.contrib.experiment.simultaneous_coherence_measurement`."""

from __future__ import annotations

from qubex.contrib import simultaneous_coherence_measurement


def test_simultaneous_coherence_measurement_is_exported_from_contrib() -> None:
    """Given contrib package, when imported, then simultaneous coherence helper is available."""
    assert callable(simultaneous_coherence_measurement)
