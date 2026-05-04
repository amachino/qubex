"""Tests for functional APIs in `qubex.contrib.experiment.thermal_excitation`."""

from __future__ import annotations

from qubex.contrib import thermal_excitation_via_rabi


def test_all_thermal_excitation_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then thermal excitation helpers are available."""
    assert callable(thermal_excitation_via_rabi)
