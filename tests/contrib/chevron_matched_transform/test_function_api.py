"""Tests for functional APIs in `qubex.contrib.experiment.chevron_matched_transform`."""

from __future__ import annotations

from qubex.contrib import (
    analyze_chevron_matched_transform,
    estimate_qubit_frequency_from_chevron,
    estimate_qubit_frequency_from_chevron_adaptive,
    measure_chevron_pattern,
)


def test_all_chevron_matched_transform_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CKP characterization helpers are available."""
    assert callable(analyze_chevron_matched_transform)
    assert callable(estimate_qubit_frequency_from_chevron)
    assert callable(estimate_qubit_frequency_from_chevron_adaptive)
    assert callable(measure_chevron_pattern)
