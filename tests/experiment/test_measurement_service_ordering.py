"""Tests for ordering helpers in measurement service."""

from __future__ import annotations

from qubex.experiment.services.measurement_service import MeasurementService


def test_unique_in_order_preserves_first_seen_order() -> None:
    """Given duplicate labels, when uniqueing, then first-seen order is preserved."""
    labels = ["Q01", "Q00", "Q01", "Q02", "Q00"]

    assert MeasurementService.unique_in_order(labels) == ["Q01", "Q00", "Q02"]


def test_ordered_qubit_labels_preserves_target_appearance() -> None:
    """Given mixed target labels, when extracting qubits, then qubit order follows first appearance."""
    labels = ["Q01_read", "Q00", "Q01-ef", "Q00-CR"]

    assert MeasurementService.ordered_qubit_labels(labels) == ["Q01", "Q00"]
