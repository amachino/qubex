"""Tests for functional APIs in `qubex.contrib.stark_characterization`."""

from __future__ import annotations

from qubex.contrib import stark_ramsey_experiment, stark_t1_experiment


def test_all_stark_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all stark helpers are available."""
    assert callable(stark_t1_experiment)
    assert callable(stark_ramsey_experiment)
