"""Tests for functional APIs in `qubex.contrib.experiment.crosstalk_cross_resonance`."""

from __future__ import annotations

from qubex.contrib import (
    cr_crosstalk_hamiltonian_tomography,
    measure_cr_crosstalk,
)


def test_all_cr_crosstalk_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CR crosstalk helpers are available."""
    assert callable(measure_cr_crosstalk)
    assert callable(cr_crosstalk_hamiltonian_tomography)
