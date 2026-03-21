"""Tests for Measurement qubit label accessors."""

from __future__ import annotations

import pytest

from qubex.measurement.measurement import Measurement


def test_qubit_labels_returns_configured_labels() -> None:
    """Given selected qubits, when qubit_labels is accessed, then configured labels are returned."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00", "Q01"],
        load_configs=False,
        connect_devices=False,
    )

    assert measurement.qubit_labels == ["Q00", "Q01"]


def test_qubits_is_deprecated_alias_of_qubit_labels() -> None:
    """Given selected qubits, when qubits is accessed, then a deprecation warning is emitted and labels are returned."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00", "Q01"],
        load_configs=False,
        connect_devices=False,
    )

    with pytest.warns(DeprecationWarning, match="qubit_labels"):
        qubits = measurement.qubits

    assert qubits == measurement.qubit_labels
