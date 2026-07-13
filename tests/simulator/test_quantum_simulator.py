"""Tests for quantum simulator solver entrypoints."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose

from qubex.simulator import Control, QuantumSimulator, QuantumSystem, Transmon


def _driven_single_qubit() -> tuple[QuantumSystem, Control]:
    qubit = Transmon(label="Q0", dimension=2, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    waveform = np.full(40, 2 * np.pi * 0.01, dtype=np.complex128)
    durations = np.full(40, 0.1)
    control = Control(
        target=qubit,
        waveform=waveform,
        durations=durations,
        frequency=qubit.frequency,
    )
    return system, control


def test_sesolve_matches_mesolve_for_closed_system() -> None:
    """Given a closed driven system, sesolve should match mesolve dynamics."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = system.state({"Q0": "0"})

    mesolve_result = simulator.mesolve(
        [control],
        initial_state=initial_state,
        dt=0.1,
    )
    sesolve_result = simulator.sesolve(
        [control],
        initial_state=initial_state,
        dt=0.1,
    )

    assert sesolve_result.final_state.isket
    assert sesolve_result.unitaries.size == 0
    assert sesolve_result.model is not None
    assert_allclose(
        qt.ket2dm(sesolve_result.final_state).full(),
        mesolve_result.final_state.full(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_sesolve_population_display_supports_ket_states(capsys) -> None:
    """Given sesolve output, population display should handle ket states."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)

    result = simulator.sesolve(
        [control],
        initial_state=system.state({"Q0": "0"}),
        dt=0.1,
    )

    result.show_last_population()

    captured = capsys.readouterr()
    assert "|0⟩:" in captured.out
    assert "|1⟩:" in captured.out


def test_sesolve_rejects_density_matrix_initial_state() -> None:
    """Given density matrix input, sesolve should require a pure ket state."""
    system, control = _driven_single_qubit()
    simulator = QuantumSimulator(system)
    initial_state = qt.ket2dm(system.ground_state)

    with pytest.raises(ValueError, match="requires a ket initial_state"):
        simulator.sesolve(
            [control],
            initial_state=initial_state,
            dt=0.1,
        )


def test_sesolve_builds_model_without_collapse_operators() -> None:
    """Given a dissipative system, sesolve should build a closed-system model."""
    qubit = Transmon(
        label="Q0",
        dimension=2,
        frequency=5.0,
        relaxation_rate=0.01,
    )
    system = QuantumSystem(objects=[qubit])
    waveform = np.full(40, 2 * np.pi * 0.01, dtype=np.complex128)
    durations = np.full(40, 0.1)
    control = Control(
        target=qubit,
        waveform=waveform,
        durations=durations,
        frequency=qubit.frequency,
    )
    simulator = QuantumSimulator(system)

    result = simulator.sesolve(
        [control],
        initial_state=system.state({"Q0": "0"}),
        dt=0.1,
    )

    assert result.final_state.isket
    assert result.model is not None
    assert result.model.collapse_operators == []
