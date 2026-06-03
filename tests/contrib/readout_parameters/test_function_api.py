"""Tests for functional APIs in `qubex.contrib.experiment.readout_parameters`."""

from __future__ import annotations

from qubex.contrib import (
    characterize_readout_parameters,
    characterize_readout_parameters_2d,
    fit_readout_parameters,
)


def test_all_readout_parameter_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then readout parameter helpers are available."""
    assert callable(characterize_readout_parameters)
    assert callable(characterize_readout_parameters_2d)
    assert callable(fit_readout_parameters)
