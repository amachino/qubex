"""Tests for compiling local quantum-system models."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose
from qxsimulator import (
    ChargeBasisEigensystem,
    CompiledCosineTransmon,
    CompiledObject,
    Qubit,
    Resonator,
    Transmon,
)


def test_qubit_compile_builds_local_operators() -> None:
    """Qubit compilation should build its local Hamiltonian and operators."""
    qubit = Qubit(
        label="Q0",
        frequency=5.0,
        relaxation_rate=0.01,
        dephasing_rate=0.002,
    )

    compiled = qubit.compile()

    lowering = qt.destroy(2)
    assert isinstance(compiled, CompiledObject)
    assert compiled.source is qubit
    assert_allclose(
        compiled.hamiltonian.full(),
        (2 * np.pi * qubit.frequency * qt.num(2)).full(),
        rtol=1e-12,
        atol=0.0,
    )
    assert compiled.lowering_operator == lowering
    assert compiled.interaction_operator == lowering + lowering.dag()
    assert len(compiled.collapse_operators) == 2
    assert_allclose(
        compiled.collapse_operators[0].full(),
        (np.sqrt(qubit.relaxation_rate) * lowering).full(),
        rtol=1e-12,
        atol=0.0,
    )
    assert_allclose(
        compiled.collapse_operators[1].full(),
        (np.sqrt(2 * qubit.dephasing_rate) * qt.num(2)).full(),
        rtol=1e-12,
        atol=0.0,
    )


def test_resonator_compile_builds_harmonic_model() -> None:
    """Resonator compilation should build a truncated harmonic model."""
    resonator = Resonator(label="R0", dimension=4, frequency=7.0)

    compiled = resonator.compile()

    assert compiled.source is resonator
    assert_allclose(
        compiled.hamiltonian.diag(),
        2 * np.pi * resonator.frequency * np.arange(resonator.dimension),
        rtol=1e-12,
        atol=0.0,
    )
    assert compiled.lowering_operator == qt.destroy(resonator.dimension)
    assert compiled.collapse_operators == ()


def test_duffing_transmon_compile_preserves_existing_spectrum() -> None:
    """Duffing transmon compilation should reproduce the analytic spectrum."""
    transmon = Transmon(
        label="Q0",
        dimension=5,
        frequency=5.0,
        anharmonicity=-0.25,
    )

    compiled = transmon.compile()

    levels = np.arange(transmon.dimension)
    expected = (
        2
        * np.pi
        * (
            transmon.frequency * levels
            + 0.5 * transmon.anharmonicity * levels * (levels - 1)
        )
    )
    assert transmon.model == "duffing"
    assert compiled.source is transmon
    assert_allclose(compiled.hamiltonian.diag(), expected, rtol=1e-12, atol=0.0)


def test_cosine_transmon_compile_uses_energy_basis_charge_operator() -> None:
    """Cosine compilation should retain the full projected charge operator."""
    transmon = Transmon(
        label="Q0",
        dimension=7,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=20,
    )

    compiled = transmon.compile()

    assert isinstance(compiled, CompiledCosineTransmon)
    assert isinstance(compiled.charge_basis, ChargeBasisEigensystem)
    energies = np.asarray(compiled.hamiltonian.diag(), dtype=float) / (2 * np.pi)
    interaction = compiled.interaction_operator.full()
    lowering = compiled.lowering_operator.full()
    assert compiled.source is transmon
    assert_allclose(energies[0], 0.0, rtol=0.0, atol=1e-12)
    assert_allclose(energies[1] - energies[0], 5.0, rtol=0.0, atol=1e-9)
    assert_allclose(
        energies[2] - 2 * energies[1] + energies[0],
        -0.25,
        rtol=0.0,
        atol=1e-9,
    )
    assert_allclose(interaction, interaction.conj().T, rtol=0.0, atol=1e-12)
    assert_allclose(interaction[0, 1], 1.0, rtol=0.0, atol=1e-12)
    adjacent_elements = np.diag(interaction, k=1)
    assert_allclose(adjacent_elements.imag, 0.0, rtol=0.0, atol=1e-12)
    assert np.all(adjacent_elements.real > 0.0)
    assert abs(interaction[0, 3]) > 1e-3
    expected_lowering = np.zeros_like(interaction)
    levels = np.arange(transmon.dimension - 1)
    expected_lowering[levels, levels + 1] = interaction[levels, levels + 1]
    assert_allclose(lowering, expected_lowering, rtol=0.0, atol=1e-12)


def test_cosine_transmon_compile_exposes_charge_basis_eigensystem() -> None:
    """Cosine compilation should expose the retained charge-basis eigensystem."""
    transmon = Transmon(
        label="Q0",
        dimension=5,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=15,
        offset_charge=0.2,
    )

    compiled = transmon.compile()

    assert isinstance(compiled, CompiledCosineTransmon)
    charge_basis = compiled.charge_basis
    assert charge_basis.offset_charge == pytest.approx(0.2, rel=0.0, abs=1e-15)
    assert charge_basis.charge_numbers.shape == (31,)
    assert charge_basis.hamiltonian.shape == (31, 31)
    assert charge_basis.eigenenergies.shape == (transmon.dimension,)
    assert charge_basis.eigenvectors.shape == (31, transmon.dimension)

    expected_charge_hamiltonian = np.diag(
        4
        * charge_basis.charging_energy
        * (charge_basis.charge_numbers - charge_basis.offset_charge) ** 2
    )
    hopping = np.full(30, -0.5 * charge_basis.josephson_energy)
    expected_charge_hamiltonian += np.diag(hopping, k=1) + np.diag(hopping, k=-1)
    assert_allclose(
        charge_basis.hamiltonian,
        expected_charge_hamiltonian,
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        charge_basis.hamiltonian @ charge_basis.eigenvectors,
        charge_basis.eigenvectors * charge_basis.eigenenergies,
        rtol=1e-10,
        atol=1e-10,
    )

    relative_charges = charge_basis.charge_numbers - charge_basis.offset_charge
    projected_charge = charge_basis.eigenvectors.conj().T @ (
        relative_charges[:, np.newaxis] * charge_basis.eigenvectors
    )
    normalized_charge = projected_charge / abs(projected_charge[0, 1])
    assert_allclose(
        compiled.interaction_operator.full(),
        normalized_charge,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("offset_charge", "equivalent_offset_charge"),
    [
        (0.0, 1.0),
        (0.2, 20.2),
        (-0.2, -20.2),
        (-0.5, 0.5),
    ],
)
def test_cosine_transmon_compile_is_periodic_in_offset_charge(
    offset_charge: float,
    equivalent_offset_charge: float,
) -> None:
    """Cosine compilation should be invariant under integer offset-charge shifts."""
    parameters = {
        "label": "Q0",
        "dimension": 7,
        "frequency": 5.0,
        "anharmonicity": -0.25,
        "model": "cosine",
        "charge_cutoff": 15,
    }

    compiled = Transmon(offset_charge=offset_charge, **parameters).compile()
    equivalent = Transmon(
        offset_charge=equivalent_offset_charge,
        **parameters,
    ).compile()

    assert isinstance(compiled, CompiledCosineTransmon)
    assert isinstance(equivalent, CompiledCosineTransmon)
    assert_allclose(
        equivalent.charge_basis.offset_charge,
        compiled.charge_basis.offset_charge,
        rtol=0.0,
        atol=1e-14,
    )
    assert_allclose(
        equivalent.charge_basis.hamiltonian,
        compiled.charge_basis.hamiltonian,
        rtol=0.0,
        atol=1e-10,
    )
    assert_allclose(
        equivalent.hamiltonian.full(),
        compiled.hamiltonian.full(),
        rtol=0.0,
        atol=1e-10,
    )
    assert_allclose(
        equivalent.interaction_operator.full(),
        compiled.interaction_operator.full(),
        rtol=0.0,
        atol=1e-10,
    )


def test_cosine_transmon_compile_keeps_relative_charge_diagonal_terms() -> None:
    """Cosine interaction operators should keep physical diagonal charge terms."""
    transmon = Transmon(
        label="Q0",
        dimension=4,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=15,
        offset_charge=0.2,
    )

    compiled = transmon.compile()

    diagonal = np.diag(compiled.interaction_operator.full())
    assert np.max(np.abs(diagonal)) > 1e-8
    assert_allclose(
        np.diag(compiled.lowering_operator.full()),
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_cosine_transmon_compile_removes_integer_offset_from_charge() -> None:
    """Integer offset charge should not add identity to the interaction operator."""
    transmon = Transmon(
        label="Q0",
        dimension=4,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=15,
        offset_charge=1.0,
    )

    compiled = transmon.compile()

    assert_allclose(
        np.diag(compiled.interaction_operator.full()),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "invalid"}, "Unsupported transmon model"),
        ({"model": "cosine", "charge_cutoff": 0}, "charge_cutoff"),
        (
            {"dimension": 8, "model": "cosine", "charge_cutoff": 2},
            "charge basis",
        ),
    ],
)
def test_transmon_rejects_invalid_compilation_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Transmon should reject invalid local-model compilation parameters."""
    parameters: dict[str, object] = {
        "label": "Q0",
        "dimension": 4,
        "frequency": 5.0,
        "anharmonicity": -0.25,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        Transmon(**parameters)  # type: ignore[arg-type]
