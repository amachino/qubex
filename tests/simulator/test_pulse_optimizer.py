"""Tests for the deprecated pulse optimizer API."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
import qutip as qt
from qxsimulator import PulseOptimizer, QuantumSystem, Transmon


def test_pulse_optimizer_warns_when_constructed() -> None:
    """Constructing PulseOptimizer should emit its deprecation warning."""
    system = QuantumSystem(
        objects=[Transmon(label="Q0", dimension=2, frequency=5.0)],
    )
    nonunitary_target = qt.Qobj(np.zeros((2, 2), dtype=np.complex128))

    with (
        pytest.warns(DeprecationWarning, match="PulseOptimizer is deprecated"),
        pytest.raises(ValueError, match="must be unitary"),
    ):
        PulseOptimizer(
            quantum_system=system,
            target_unitary=nonunitary_target,
            initial_state=system.ground_state,
            control_frequencies={"Q0": 5.0},
            segment_count=1,
            segment_width=1.0,
            max_rabi_frequency=0.1,
        )


def test_pulse_optimizer_reports_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing without JAX should explain the compatibility requirement."""
    optimizer_module = import_module("qxsimulator.optimization.pulse_optimizer")
    original_import_module = optimizer_module.importlib.import_module

    def import_without_optimizer_dependencies(name: str) -> object:
        if name.partition(".")[0] in {"jax", "optax"}:
            raise ModuleNotFoundError(name)
        return original_import_module(name)

    monkeypatch.setattr(
        optimizer_module.importlib,
        "import_module",
        import_without_optimizer_dependencies,
    )
    system = QuantumSystem(
        objects=[Transmon(label="Q0", dimension=2, frequency=5.0)],
    )

    with (
        pytest.warns(DeprecationWarning, match="PulseOptimizer is deprecated"),
        pytest.raises(
            ModuleNotFoundError,
            match="no longer installed with qxsimulator",
        ),
    ):
        PulseOptimizer(
            quantum_system=system,
            target_unitary=qt.qeye(2),
            initial_state=system.ground_state,
            control_frequencies={"Q0": 5.0},
            segment_count=1,
            segment_width=1.0,
            max_rabi_frequency=0.1,
        )
