"""Tests for MCM randomized benchmarking public exports."""

from __future__ import annotations

from qubex.contrib import mcm_randomized_benchmarking, mcm_rb_sequence


def test_mcm_randomized_benchmarking_functions_are_exported() -> None:
    """The contrib package should export the sequence and experiment helpers."""
    assert callable(mcm_rb_sequence)
    assert callable(mcm_randomized_benchmarking)
