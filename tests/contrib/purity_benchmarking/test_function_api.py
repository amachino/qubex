"""Tests for functional APIs in `qubex.contrib.experiment.purity_benchmarking`."""

from __future__ import annotations

from qubex.contrib import (
    interleaved_purity_benchmarking,
    ipb_experiment,
    pb_experiment_1q,
    pb_experiment_2q,
    purity_benchmarking,
    purity_sequence_1q,
    purity_sequence_2q,
)


def test_all_purity_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all purity helpers are available."""
    assert callable(purity_sequence_1q)
    assert callable(purity_sequence_2q)
    assert callable(pb_experiment_1q)
    assert callable(pb_experiment_2q)
    assert callable(ipb_experiment)
    assert callable(purity_benchmarking)
    assert callable(interleaved_purity_benchmarking)
