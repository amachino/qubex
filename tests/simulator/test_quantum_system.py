"""Tests for quantum-system container behavior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose
from qxsimulator import Coupling, QuantumSystem, Qubit, Resonator, Transmon, gates


def create_system() -> QuantumSystem:
    """Create a two-qubit system for graph-view tests."""
    qubit_0 = Qubit(label="Q0", frequency=5.0)
    qubit_1 = Qubit(label="Q1", frequency=5.2)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.005)
    return QuantumSystem(objects=[qubit_0, qubit_1], couplings=[coupling])


def test_quantum_system_builds_internal_lookup_maps() -> None:
    """QuantumSystem should index objects by label and couplings by pair."""
    system = create_system()
    qubit_0, qubit_1 = system.objects
    (coupling,) = system.couplings

    assert system._objects_by_label == {  # noqa: SLF001
        "Q0": qubit_0,
        "Q1": qubit_1,
    }
    assert system._object_indices_by_label == {  # noqa: SLF001
        "Q0": 0,
        "Q1": 1,
    }
    assert system._couplings_by_pair == {  # noqa: SLF001
        ("Q0", "Q1"): coupling,
    }


def test_quantum_system_snapshots_input_sequences() -> None:
    """QuantumSystem should not change when its input lists are mutated."""
    qubit_0 = Qubit(label="Q0", frequency=5.0)
    qubit_1 = Qubit(label="Q1", frequency=5.2)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.005)
    objects = [qubit_0, qubit_1]
    couplings = [coupling]
    system = QuantumSystem(objects=objects, couplings=couplings)

    objects.clear()
    couplings.clear()

    assert system.objects == (qubit_0, qubit_1)
    assert system.couplings == (coupling,)
    assert system.get_object("Q0") is qubit_0
    assert system.get_coupling(("Q1", "Q0")) is coupling


def test_graph_returns_a_fresh_representation() -> None:
    """Graph mutations should not change the system topology."""
    system = create_system()

    graph = system.graph
    graph.remove_edge("Q0", "Q1")

    assert system.graph is not graph
    assert system.graph.has_edge("Q0", "Q1")
    assert system.get_coupled_objects("Q0") == [system.get_object("Q1")]


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("object_labels", ["Q0", "Q1"]),
        ("object_dimensions", [2, 2]),
        ("coupling_labels", ["Q0-Q1"]),
    ],
)
def test_metadata_lists_return_fresh_values(
    attribute: str,
    expected: list[str] | list[int],
) -> None:
    """Metadata-list mutations should not change later property values."""
    system = create_system()

    value = getattr(system, attribute)
    assert isinstance(value, list)
    value.clear()

    assert getattr(system, attribute) == expected


def test_ground_state_returns_a_fresh_qobj() -> None:
    """Ground-state callers should receive independent Qobj instances."""
    system = create_system()

    state = system.ground_state

    assert system.ground_state is not state
    assert system.ground_state == state


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("node_set", r"use set\(object_labels\) instead"),
        ("edge_set", r"use set\(graph.edges\) instead"),
        ("node_list", "use object_labels instead"),
        ("edge_list", r"use list\(graph.edges\) instead"),
    ],
)
def test_graph_view_properties_are_deprecated(attribute: str, message: str) -> None:
    """Legacy graph-view properties should warn on every access."""
    system = create_system()

    with pytest.warns(DeprecationWarning, match=message):
        getattr(system, attribute)

    with pytest.warns(DeprecationWarning, match=message):
        getattr(system, attribute)


def test_number_matrix_is_deprecated() -> None:
    """Number matrix should warn on every access."""
    system = create_system()
    message = r"use get_number_operator\(label\) for local number operators"

    with pytest.warns(DeprecationWarning, match=message):
        number_matrix = system.number_matrix

    with pytest.warns(DeprecationWarning, match=message):
        assert system.number_matrix == number_matrix


def test_internal_operations_do_not_use_deprecated_graph_views() -> None:
    """Internal operations should not access deprecated graph-view properties."""
    system = create_system()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert system.get_index("Q0") == 0
        assert system.get_object("Q0").label == "Q0"
        assert system.get_lowering_operator("Q0").shape == (4, 4)
        assert system.state({"Q0": 0}).shape == (4, 1)
        assert system.get_coupled_objects("Q0") == [system.get_object("Q1")]


def test_quantum_system_embeds_compiled_local_operators() -> None:
    """QuantumSystem should embed operators compiled by each source object."""
    transmon = Transmon(
        label="Q0",
        dimension=5,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=15,
        relaxation_rate=0.01,
        dephasing_rate=0.002,
    )
    resonator = Resonator(label="R0", dimension=3, frequency=7.0)
    system = QuantumSystem(objects=[transmon, resonator])
    compiled = transmon.compile()
    identity = qt.qeye(resonator.dimension)

    assert_allclose(
        system.get_object_hamiltonian(transmon.label).full(),
        qt.tensor(compiled.hamiltonian, identity).full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        system.get_lowering_operator(transmon.label).full(),
        qt.tensor(compiled.lowering_operator, identity).full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        system.get_interaction_operator(transmon.label).full(),
        qt.tensor(compiled.interaction_operator, identity).full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        system.get_number_operator(transmon.label).full(),
        qt.tensor(qt.num(transmon.dimension), identity).full(),
        rtol=0.0,
        atol=1e-12,
    )
    for actual, local in zip(
        system.get_collapse_operators(transmon.label),
        compiled.collapse_operators,
        strict=True,
    ):
        assert_allclose(
            actual.full(),
            qt.tensor(local, identity).full(),
            rtol=0.0,
            atol=1e-12,
        )


def test_truncate_operator_restricts_each_object_to_qubit_levels() -> None:
    """Operator truncation should retain the tensor-product qubit subspace."""
    qubit_0 = Transmon(label="Q0", dimension=3, frequency=5.0)
    qubit_1 = Transmon(label="Q1", dimension=3, frequency=5.2)
    system = QuantumSystem(objects=[qubit_0, qubit_1])
    operator = qt.tensor(qt.qeye(3), qt.qeye(3))

    truncated = system.truncate_operator(operator)

    assert truncated.dims == [[2, 2], [2, 2]]
    assert_allclose(truncated.full(), np.eye(4), rtol=0.0, atol=0.0)


def test_unitary_embeds_named_gates_in_system_order() -> None:
    """Named gates should embed by label while preserving target orientation."""
    q01 = Transmon(label="Q01", dimension=3, frequency=5.0)
    q04 = Transmon(label="Q04", dimension=3, frequency=5.2)
    resonator = Resonator(label="R0", dimension=2, frequency=7.0)
    system = QuantumSystem(objects=[q01, q04, resonator])
    unitary = system.unitary({"Q04-Q01": "CNOT"})

    actual = unitary @ system.state({"Q04": 1, "Q01": 0, "R0": 1})
    expected = system.state({"Q04": 1, "Q01": 1, "R0": 1})

    assert unitary.dims == [[3, 3, 2], [3, 3, 2]]
    assert unitary.isunitary
    assert_allclose(actual.full(), expected.full(), rtol=0.0, atol=1e-12)


def test_unitary_is_identity_outside_embedded_gate_levels() -> None:
    """A qubit gate embedded in a qutrit should leave the f level unchanged."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])

    unitary = system.unitary({"Q0": "X"})

    assert_allclose(
        (unitary @ system.state({"Q0": 2})).full(),
        system.state({"Q0": 2}).full(),
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_combines_disjoint_local_gates() -> None:
    """Disjoint local operations should form a tensor product in system order."""
    q0 = Qubit(label="Q0", frequency=5.0)
    q1 = Qubit(label="Q1", frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])

    unitary = system.unitary({"Q1": "H", "Q0": "X"})

    assert_allclose(
        unitary.full(),
        qt.tensor(gates.X, gates.H).full(),
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_accepts_full_qudit_gate() -> None:
    """A custom qudit gate should embed with its declared local dimension."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])
    cyclic_shift = qt.Qobj(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )

    unitary = system.unitary({"Q0": cyclic_shift})

    assert_allclose(unitary.full(), cyclic_shift.full(), rtol=0.0, atol=0.0)


def test_unitary_embeds_gate_on_explicit_levels() -> None:
    """Explicit levels should select the local subspace used for embedding."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])

    unitary = system.unitary({"Q0": gates.X}, levels={"Q0": (1, 2)})

    assert_allclose(
        (unitary @ system.state({"Q0": 1})).full(),
        system.state({"Q0": 2}).full(),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        (unitary @ system.state({"Q0": 0})).full(),
        system.state({"Q0": 0}).full(),
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_accepts_tuple_targets_for_oriented_gates() -> None:
    """Tuple targets should preserve the tensor-factor order of oriented gates."""
    q0 = Qubit(label="Q0", frequency=5.0)
    q1 = Qubit(label="Q1", frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])

    unitary = system.unitary({("Q1", "Q0"): "CNOT"})
    actual = unitary @ system.state({"Q1": 1, "Q0": 0})

    assert_allclose(
        actual.full(),
        system.state({"Q1": 1, "Q0": 1}).full(),
        rtol=0.0,
        atol=1e-12,
    )


def test_unitary_rejects_overlapping_gate_targets() -> None:
    """A parallel unitary specification should reject overlapping operations."""
    q0 = Qubit(label="Q0", frequency=5.0)
    q1 = Qubit(label="Q1", frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])

    with pytest.raises(ValueError, match=r"Q1.*more than one gate"):
        system.unitary({("Q0", "Q1"): "CZ", "Q1": "X"})


def test_unitary_rejects_gate_arity_mismatch() -> None:
    """Gate tensor factors should match the number of target objects."""
    q0 = Qubit(label="Q0", frequency=5.0)
    q1 = Qubit(label="Q1", frequency=5.2)
    system = QuantumSystem(objects=[q0, q1])

    with pytest.raises(ValueError, match=r"2 tensor factors.*1 target"):
        system.unitary({"Q0": "CZ"})


def test_unitary_rejects_invalid_embedding_levels() -> None:
    """Embedding levels should be unique, in range, and match the gate size."""
    qubit = Transmon(label="Q0", dimension=3, frequency=5.0)
    system = QuantumSystem(objects=[qubit])

    with pytest.raises(ValueError, match="exactly 2 levels"):
        system.unitary({"Q0": "X"}, levels={"Q0": (0, 1, 2)})
    with pytest.raises(ValueError, match="unique"):
        system.unitary({"Q0": "X"}, levels={"Q0": (1, 1)})
    with pytest.raises(ValueError, match=r"outside.*Q0"):
        system.unitary({"Q0": "X"}, levels={"Q0": (1, 3)})


def test_rotating_object_hamiltonian_removes_the_reference_frequency() -> None:
    """Rotating local Hamiltonians should remove frequency times level number."""
    transmon = Transmon(
        label="Q0",
        dimension=5,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=15,
    )
    system = QuantumSystem(objects=[transmon])

    rotating = system.get_rotating_object_hamiltonian(transmon.label)

    expected = transmon.compile().hamiltonian - (
        2 * np.pi * transmon.frequency * qt.num(transmon.dimension)
    )
    assert_allclose(rotating.full(), expected.full(), rtol=0.0, atol=1e-12)


def test_perturbative_static_zz_returns_full_zeta() -> None:
    """Perturbative static ZZ should return the full conditional splitting."""
    transmon_0 = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        anharmonicity=-0.25,
    )
    transmon_1 = Transmon(
        label="Q1",
        dimension=3,
        frequency=5.7,
        anharmonicity=-0.30,
    )
    coupling = Coupling(pair=(transmon_0, transmon_1), strength=0.02)
    system = QuantumSystem(
        objects=[transmon_0, transmon_1],
        couplings=[coupling],
    )
    detuning = transmon_0.frequency - transmon_1.frequency
    expected = (
        2
        * coupling.strength**2
        * (transmon_0.anharmonicity + transmon_1.anharmonicity)
        / (
            (detuning + transmon_0.anharmonicity)
            * (detuning - transmon_1.anharmonicity)
        )
    )

    assert system.get_static_zz(("Q0", "Q1")) == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_numerical_static_zz_returns_the_full_dressed_energy_combination() -> None:
    """Numerical static ZZ should equal the full dressed energy combination."""
    transmon_0 = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        anharmonicity=-0.25,
    )
    transmon_1 = Transmon(
        label="Q1",
        dimension=3,
        frequency=5.7,
        anharmonicity=-0.30,
    )
    coupling = Coupling(pair=(transmon_0, transmon_1), strength=0.02)
    system = QuantumSystem(
        objects=[transmon_0, transmon_1],
        couplings=[coupling],
    )

    two_excitation_hamiltonian = np.array(
        [
            [
                2 * transmon_0.frequency + transmon_0.anharmonicity,
                np.sqrt(2) * coupling.strength,
                0.0,
            ],
            [
                np.sqrt(2) * coupling.strength,
                transmon_0.frequency + transmon_1.frequency,
                np.sqrt(2) * coupling.strength,
            ],
            [
                0.0,
                np.sqrt(2) * coupling.strength,
                2 * transmon_1.frequency + transmon_1.anharmonicity,
            ],
        ]
    )
    dressed_energies, dressed_states = np.linalg.eigh(two_excitation_hamiltonian)
    dressed_11 = dressed_energies[np.argmax(np.abs(dressed_states[1, :]) ** 2)]
    expected = dressed_11 - transmon_0.frequency - transmon_1.frequency

    assert system.get_static_zz(("Q0", "Q1"), method="numerical") == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_static_zz_uses_the_compiled_cosine_pair_hamiltonian() -> None:
    """Static ZZ should use dressed energies from compiled cosine models."""
    transmon_0 = Transmon(
        label="Q0",
        dimension=4,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
        charge_cutoff=12,
    )
    transmon_1 = Transmon(
        label="Q1",
        dimension=4,
        frequency=5.7,
        anharmonicity=-0.30,
        model="cosine",
        charge_cutoff=12,
    )
    system = QuantumSystem(
        objects=[transmon_0, transmon_1],
        couplings=[Coupling(pair=(transmon_0, transmon_1), strength=0.02)],
    )
    dressed_energies, dressed_states = system.hamiltonian.eigenstates()
    bare_states = [
        system.state({"Q0": level_0, "Q1": level_1})
        for level_0, level_1 in [(0, 0), (1, 0), (0, 1), (1, 1)]
    ]
    dressed_indices = [
        int(np.argmax([abs(bare.overlap(dressed)) ** 2 for dressed in dressed_states]))
        for bare in bare_states
    ]
    assert len(set(dressed_indices)) == len(dressed_indices)
    energy_00, energy_10, energy_01, energy_11 = (
        dressed_energies[index] / (2 * np.pi) for index in dressed_indices
    )
    expected = energy_11 - energy_10 - energy_01 + energy_00

    assert system.get_static_zz(("Q0", "Q1"), method="numerical") == pytest.approx(
        expected,
        rel=1e-10,
        abs=1e-12,
    )


def test_frequency_shift_uses_half_the_full_static_zz() -> None:
    """Frequency shift should add half the full static-ZZ splitting."""
    transmon_0 = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        anharmonicity=-0.25,
    )
    transmon_1 = Transmon(
        label="Q1",
        dimension=3,
        frequency=5.7,
        anharmonicity=-0.30,
    )
    system = QuantumSystem(
        objects=[transmon_0, transmon_1],
        couplings=[Coupling(pair=(transmon_0, transmon_1), strength=0.02)],
    )

    expected = system.get_lamb_shift(("Q0", "Q1")) + 0.5 * system.get_static_zz(
        ("Q0", "Q1")
    )

    assert system.get_frequency_shift("Q0") == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_numerical_lamb_shift_uses_the_dressed_ground_state_transition() -> None:
    """Numerical Lamb shift should use the dressed transition with its neighbor in zero."""
    qubit_0 = Qubit(label="Q0", frequency=5.0)
    qubit_1 = Qubit(label="Q1", frequency=5.7)
    coupling = Coupling(pair=(qubit_0, qubit_1), strength=0.02)
    system = QuantumSystem(objects=[qubit_0, qubit_1], couplings=[coupling])
    one_excitation_hamiltonian = np.array(
        [
            [qubit_0.frequency, coupling.strength],
            [coupling.strength, qubit_1.frequency],
        ]
    )
    dressed_energies, dressed_states = np.linalg.eigh(one_excitation_hamiltonian)
    dressed_10 = dressed_energies[np.argmax(np.abs(dressed_states[0, :]) ** 2)]
    expected = dressed_10 - qubit_0.frequency

    assert system.get_lamb_shift(("Q0", "Q1"), method="numerical") == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_numerical_shift_options_propagate_to_effective_frequency() -> None:
    """Numerical shift selection should propagate to the effective frequency."""
    transmon_0 = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        anharmonicity=-0.25,
    )
    transmon_1 = Transmon(
        label="Q1",
        dimension=3,
        frequency=5.7,
        anharmonicity=-0.30,
    )
    system = QuantumSystem(
        objects=[transmon_0, transmon_1],
        couplings=[Coupling(pair=(transmon_0, transmon_1), strength=0.02)],
    )
    expected_shift = system.get_lamb_shift(
        ("Q0", "Q1"),
        method="numerical",
    ) + 0.5 * system.get_static_zz(
        ("Q0", "Q1"),
        method="numerical",
    )

    assert system.get_frequency_shift("Q0", method="numerical") == pytest.approx(
        expected_shift,
        rel=1e-12,
        abs=1e-12,
    )
    assert system.get_effective_frequency("Q0", method="numerical") == pytest.approx(
        transmon_0.frequency + expected_shift,
        rel=1e-12,
        abs=1e-12,
    )
