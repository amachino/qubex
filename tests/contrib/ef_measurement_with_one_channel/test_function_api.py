"""Tests for functional APIs in `qubex.contrib.experiment.ef_measurement_with_one_channel`."""

from __future__ import annotations

from qubex.contrib import (
    calibrate_cr_pi_pulse,
    ef_chevron_pattern,
    ef_rabi_experiment,
)


def test_all_ef_measurement_with_one_channel_functions_are_exported_from_contrib() -> (
    None
):
    """Given contrib package, when imported, then EF measurement helpers are available."""
    assert callable(calibrate_cr_pi_pulse)
    assert callable(ef_rabi_experiment)
    assert callable(ef_chevron_pattern)
