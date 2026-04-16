"""Tests for functional APIs in `qubex.contrib.experiment.readout_parameters`."""

from __future__ import annotations

from qubex.contrib import (
    characterize_readout_parameters,
)


def test_all_cr_crosstalk_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CR crosstalk helpers are available."""
    assert callable(characterize_readout_parameters)
