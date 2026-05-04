"""Tests for functional APIs in `qubex.contrib.experiment.superconducting_gap`."""

from __future__ import annotations

from qubex.contrib import get_resistance_charge, get_superconducting_gap


def test_superconducting_gap_function_is_exported_from_contrib() -> None:
    """Given contrib package, when imported, then superconducting-gap helper is available."""
    assert callable(get_superconducting_gap)
    assert callable(get_resistance_charge)
