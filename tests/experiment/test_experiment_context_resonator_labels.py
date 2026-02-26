"""Tests for ExperimentContext resonator label selection."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.experiment.experiment_context import ExperimentContext


def test_resonator_labels_follow_active_qubit_order(monkeypatch) -> None:
    """Given active qubits, when resonator_labels is accessed, then matching resonators are returned in qubit order."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_qubits"] = ["Q01", "Q00", "Q02"]
    resonators = {
        "Q00": SimpleNamespace(label="R00"),
        "Q01": SimpleNamespace(label="R01"),
    }
    monkeypatch.setattr(
        ExperimentContext,
        "resonators",
        property(lambda self: resonators),
    )

    assert context.resonator_labels == ["R01", "R00"]
