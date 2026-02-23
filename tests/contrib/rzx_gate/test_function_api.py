"""Tests for functional APIs in `qubex.contrib.experiment.rzx_gate`."""

from __future__ import annotations

from qubex.contrib import rzx, rzx_gate_property


def test_all_rzx_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then RZX helpers are available."""
    assert callable(rzx)
    assert callable(rzx_gate_property)
