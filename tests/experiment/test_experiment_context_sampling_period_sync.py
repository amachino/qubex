"""Tests for sampling-period ownership boundaries in experiment context."""

from __future__ import annotations

from qubex.experiment.experiment_context import ExperimentContext


def test_context_does_not_expose_sampling_period_sync_methods() -> None:
    """Given context class, when checking sync APIs, then sampling-period sync methods are not exposed."""
    assert not hasattr(ExperimentContext, "sync_pulse_sampling_period")
    assert not hasattr(ExperimentContext, "_sync_pulse_sampling_period")
