"""Tests for paired interleaved randomized-benchmarking exports."""

from __future__ import annotations

from qubex.contrib import (
    analyze_paired_irb,
    measure_paired_irb,
    paired_interleaved_randomized_benchmarking,
)


def test_paired_irb_functions_are_exported() -> None:
    """The contrib package should export measurement, analysis, and composed APIs."""
    assert callable(measure_paired_irb)
    assert callable(analyze_paired_irb)
    assert callable(paired_interleaved_randomized_benchmarking)
