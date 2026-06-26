"""Tests for functional APIs in `qubex.contrib.experiment.efh_ramsey_experiment`."""

from __future__ import annotations

from qubex.contrib import ef_ramsey_experiment, fh_ramsey_experiment
from qubex.contrib.experiment import (
    ef_ramsey_experiment as experiment_ef_ramsey_experiment,
    fh_ramsey_experiment as experiment_fh_ramsey_experiment,
)


def test_all_efh_ramsey_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then EF/FH Ramsey helpers are available."""
    assert callable(ef_ramsey_experiment)
    assert callable(fh_ramsey_experiment)


def test_all_efh_ramsey_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then it exposes the same helpers."""
    assert experiment_ef_ramsey_experiment is ef_ramsey_experiment
    assert experiment_fh_ramsey_experiment is fh_ramsey_experiment
