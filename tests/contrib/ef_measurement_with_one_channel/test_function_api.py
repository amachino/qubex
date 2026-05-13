"""Tests for functional APIs in `qubex.contrib.experiment.ef_measurement_with_one_channel`."""

from __future__ import annotations

from qubex.contrib import (
    calibrate_cr_pi_pulse,
    obtain_anharmonicity_with_cr,
)


def test_all_ef_measurement_with_one_channel_functions_are_exported_from_contrib() -> (
    None
):
    """Given contrib package, when imported, then EF measurement helpers are available."""
    assert callable(calibrate_cr_pi_pulse)
    assert callable(obtain_anharmonicity_with_cr)
