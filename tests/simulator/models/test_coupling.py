"""Tests for the lightweight coupling model."""

from __future__ import annotations

import pytest
from qxcore import units
from qxsimulator import Coupling, Transmon


def test_coupling_normalizes_tunits_strength_to_ghz() -> None:
    """Coupling should normalize tunits strength to a GHz float."""
    coupling = Coupling(pair=("Q0", "Q1"), strength=5 * units.MHz)

    assert coupling.strength == pytest.approx(0.005, rel=1e-12, abs=0.0)
    assert isinstance(coupling.strength, float)


def test_coupling_accepts_normalized_model_objects() -> None:
    """Coupling should accept model objects initialized with tunits values."""
    qubit_0 = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0 * units.GHz,
    )
    qubit_1 = Transmon(
        label="Q1",
        dimension=3,
        frequency=5.2 * units.GHz,
    )

    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.005)

    assert coupling.pair == ("Q0", "Q1")
    assert coupling.strength == pytest.approx(0.005, rel=1e-12, abs=0.0)
