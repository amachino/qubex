"""Tests for functional APIs in `qubex.contrib.experiment.measure_efh_chevron_pattern`."""

from __future__ import annotations

from qubex.contrib import (
    estimate_ef_frequency_from_chevron,
    estimate_ef_frequency_from_chevron_adaptive,
    estimate_fh_frequency_from_chevron,
    estimate_fh_frequency_from_chevron_adaptive,
)
from qubex.contrib.experiment import (
    estimate_ef_frequency_from_chevron as experiment_estimate_ef_frequency,
    estimate_ef_frequency_from_chevron_adaptive as experiment_estimate_ef_frequency_adaptive,
    estimate_fh_frequency_from_chevron as experiment_estimate_fh_frequency,
    estimate_fh_frequency_from_chevron_adaptive as experiment_estimate_fh_frequency_adaptive,
)


def test_all_efh_chevron_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then EF/FH chevron helpers are available."""
    assert callable(estimate_ef_frequency_from_chevron)
    assert callable(estimate_ef_frequency_from_chevron_adaptive)
    assert callable(estimate_fh_frequency_from_chevron)
    assert callable(estimate_fh_frequency_from_chevron_adaptive)


def test_all_efh_chevron_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then it exposes the same helpers."""
    assert experiment_estimate_ef_frequency is estimate_ef_frequency_from_chevron
    assert (
        experiment_estimate_ef_frequency_adaptive
        is estimate_ef_frequency_from_chevron_adaptive
    )
    assert experiment_estimate_fh_frequency is estimate_fh_frequency_from_chevron
    assert (
        experiment_estimate_fh_frequency_adaptive
        is estimate_fh_frequency_from_chevron_adaptive
    )
