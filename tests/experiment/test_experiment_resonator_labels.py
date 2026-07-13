"""Tests for Experiment resonator label accessors."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.experiment.experiment import Experiment


def test_resonator_labels_returns_context_labels() -> None:
    """Given context resonator labels, when resonator_labels is accessed, then labels are returned."""
    exp = object.__new__(Experiment)
    exp.__dict__["_experiment_context"] = SimpleNamespace(
        resonator_labels=["R00", "R01"]
    )

    assert exp.resonator_labels == ["R00", "R01"]
