"""Tests for named simulator gates."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from numpy.testing import assert_allclose
from qxsimulator import gates

from qubex.clifford import CliffordGenerator


def _pauli(name: str) -> qt.Qobj:
    operators = {
        "I": qt.qeye(2),
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
    }
    return qt.tensor(*(operators[symbol] for symbol in name))


@pytest.mark.parametrize("name", CliffordGenerator.cliffords)
def test_named_clifford_gate_matches_qubex_pauli_map(name: str) -> None:
    """Named gates should follow the Pauli-map convention used by Qubex."""
    unitary = gates.get(name)
    clifford = CliffordGenerator.cliffords[name]

    for pauli_name, mapped_pauli in clifford.map.items():
        actual = unitary @ _pauli(pauli_name) @ unitary.dag()
        expected = mapped_pauli.coefficient * _pauli(mapped_pauli.operator)
        assert_allclose(actual.full(), expected.full(), rtol=0.0, atol=1e-12)


def test_gate_lookup_is_case_insensitive_and_returns_a_copy() -> None:
    """Gate lookup should accept lowercase names without exposing registry state."""
    first = gates.get("sqrt_bswap")
    second = gates.get("SQRT_BSWAP")

    assert first is not second
    assert_allclose(first.full(), second.full(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", ["SDG", "TDG", "SXDG"])
def test_derived_inverse_gates_are_not_public_constants(name: str) -> None:
    """Derived inverse gates should not duplicate dagger and rotation operations."""
    assert name not in gates.names()
    assert not hasattr(gates, name)


@pytest.mark.parametrize(
    ("rotation_name", "pauli_name"),
    [("X180", "X"), ("Y180", "Y"), ("Z180", "Z")],
)
def test_full_pauli_rotations_reuse_named_pauli_gates(
    rotation_name: str,
    pauli_name: str,
) -> None:
    """Full Pauli rotations should be lookup aliases, not duplicate constants."""
    assert not hasattr(gates, rotation_name)
    assert_allclose(
        gates.get(rotation_name).full(),
        gates.get(pauli_name).full(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (gates.H, qt.gates.snot()),
        (gates.S, qt.gates.s_gate()),
        (gates.T, qt.gates.t_gate()),
        (gates.SX, qt.gates.sqrtnot()),
        (gates.X90, qt.gates.rx(np.pi / 2)),
        (gates.Y90, qt.gates.ry(np.pi / 2)),
        (gates.Z90, qt.gates.rz(np.pi / 2)),
        (gates.CNOT, qt.gates.cnot()),
        (gates.CY, qt.gates.cy_gate()),
        (gates.CZ, qt.gates.cz_gate()),
        (gates.SWAP, qt.gates.swap()),
        (gates.ISWAP, qt.gates.iswap()),
        (gates.SQRT_ISWAP, qt.gates.sqrtiswap()),
    ],
)
def test_standard_gates_match_qutip(
    actual: qt.Qobj,
    expected: qt.Qobj,
) -> None:
    """Standard gate constants should preserve QuTiP's matrix conventions."""
    assert_allclose(actual.full(), expected.full(), rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (
            gates.ZX90,
            np.array(
                [
                    [1, -1j, 0, 0],
                    [-1j, 1, 0, 0],
                    [0, 0, 1, 1j],
                    [0, 0, 1j, 1],
                ]
            )
            / np.sqrt(2),
        ),
        (
            gates.ZZ90,
            np.diag(
                [
                    np.exp(-0.25j * np.pi),
                    np.exp(0.25j * np.pi),
                    np.exp(0.25j * np.pi),
                    np.exp(-0.25j * np.pi),
                ]
            ),
        ),
        (
            gates.BSWAP,
            np.array(
                [
                    [0, 0, 0, 1j],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1j, 0, 0, 0],
                ]
            ),
        ),
        (
            gates.SQRT_BSWAP,
            np.array(
                [
                    [1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2)],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1j / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                ]
            ),
        ),
    ],
)
def test_fixed_rotation_gates_match_closed_form(
    actual: qt.Qobj,
    expected: np.ndarray,
) -> None:
    """Fixed rotation gates should preserve their signed matrix convention."""
    assert_allclose(actual.full(), expected, rtol=0.0, atol=1e-12)


def test_square_root_bswap_composes_to_bswap() -> None:
    """SQRT_BSWAP should square to BSWAP with the Qubex phase convention."""
    actual = gates.SQRT_BSWAP @ gates.SQRT_BSWAP

    assert_allclose(actual.full(), gates.BSWAP.full(), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("generator", [gates.X, gates.Y, gates.Z, gates.ZX])
def test_rotation_uses_generator_and_angle(generator: qt.Qobj) -> None:
    """Rotation should return exp(-i angle generator / 2)."""
    angle = 0.37

    actual = gates.rotation(generator, angle)
    expected = (-0.5j * angle * generator).expm()

    assert actual.isunitary
    assert_allclose(actual.full(), expected.full(), rtol=0.0, atol=1e-12)


def test_two_qubit_generators_are_pauli_products() -> None:
    """Named two-qubit generators should preserve their tensor-factor order."""
    assert_allclose(
        gates.XX.full(),
        qt.tensor(gates.X, gates.X).full(),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        gates.YY.full(),
        qt.tensor(gates.Y, gates.Y).full(),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        gates.ZZ.full(),
        qt.tensor(gates.Z, gates.Z).full(),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        gates.ZX.full(),
        qt.tensor(gates.Z, gates.X).full(),
        rtol=0.0,
        atol=0.0,
    )


def test_bswap_is_generated_by_yy_minus_xx() -> None:
    """The YY-minus-XX generator should reproduce the Qubex bSWAP phase."""
    angle = 0.41

    actual = gates.rotation((gates.YY - gates.XX) / 2, angle)

    assert actual.isunitary
    assert_allclose(
        actual.full(),
        np.array(
            [
                [np.cos(angle / 2), 0, 0, 1j * np.sin(angle / 2)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1j * np.sin(angle / 2), 0, 0, np.cos(angle / 2)],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_iswap_is_generated_by_negative_xx_plus_yy() -> None:
    """The negative exchange generator should reproduce the Qubex iSWAP phase."""
    actual = gates.rotation(-(gates.XX + gates.YY) / 2, np.pi)

    assert_allclose(actual.full(), gates.ISWAP.full(), rtol=0.0, atol=1e-12)


def test_rotation_rejects_non_hermitian_generator() -> None:
    """Rotation should reject an operator that cannot generate a unitary."""
    with pytest.raises(ValueError, match="Hermitian"):
        gates.rotation(qt.destroy(2), np.pi / 2)


def test_rotation_rejects_non_qobj_generator() -> None:
    """Rotation should require a dimension-carrying QuTiP operator."""
    with pytest.raises(TypeError, match="Qobj"):
        gates.rotation(np.eye(2), np.pi / 2)  # type: ignore[arg-type]


@pytest.mark.parametrize("angle", [np.nan, np.inf, -np.inf])
def test_rotation_rejects_nonfinite_angle(angle: float) -> None:
    """Rotation should reject nonfinite angles."""
    with pytest.raises(ValueError, match="finite"):
        gates.rotation(gates.X, angle)


def test_unknown_gate_name_reports_available_names() -> None:
    """Unknown gate names should fail with a discoverable error."""
    with pytest.raises(ValueError, match=r"Unknown gate 'NOPE'.*X90.*ZX90"):
        gates.get("NOPE")
