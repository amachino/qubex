"""Tests for model integration with the quantum-system container."""

from __future__ import annotations

import pytest
from qxcore import units
from qxsimulator import Coupling, QuantumSystem, Qubit, Resonator
from qxsimulator.system.models import Object


@pytest.mark.parametrize(
    "model",
    [
        Qubit(label="Q0", frequency=5 * units.GHz),
        Resonator(label="R0", dimension=5, frequency=7 * units.GHz),
    ],
)
def test_quantum_system_returns_imported_model_types(model: Object) -> None:
    """QuantumSystem should return its original model objects."""
    system = QuantumSystem(objects=[model])

    assert system.get_object(model.label) is model


@pytest.mark.parametrize(
    "label",
    [
        ("Q0", "Q1"),
        ("Q1", "Q0"),
        "Q0-Q1",
        "Q1-Q0",
    ],
)
def test_quantum_system_returns_original_coupling(
    label: str | tuple[str, str],
) -> None:
    """QuantumSystem should return the original coupling instance."""
    qubit_0 = Qubit(label="Q0", frequency=5 * units.GHz)
    qubit_1 = Qubit(label="Q1", frequency=5.2 * units.GHz)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=5 * units.MHz)
    system = QuantumSystem(
        objects=[qubit_0, qubit_1],
        couplings=[coupling],
    )

    assert system.get_coupling(label) is coupling


@pytest.mark.parametrize(
    "duplicate_pair",
    [
        ("Q0", "Q1"),
        ("Q1", "Q0"),
    ],
)
def test_quantum_system_rejects_duplicate_coupling_pairs(
    duplicate_pair: tuple[str, str],
) -> None:
    """QuantumSystem should reject duplicate pairs regardless of orientation."""
    qubit_0 = Qubit(label="Q0", frequency=5 * units.GHz)
    qubit_1 = Qubit(label="Q1", frequency=5.2 * units.GHz)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=5 * units.MHz)
    duplicate = Coupling(pair=duplicate_pair, strength=6 * units.MHz)

    with pytest.raises(ValueError, match="unique object pairs"):
        QuantumSystem(
            objects=[qubit_0, qubit_1],
            couplings=[coupling, duplicate],
        )
