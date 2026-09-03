"""Public state-tomography regressions for complex coherence conventions."""

from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray

from qubex.analysis.state_tomography import (
    calculate_expected_values,
    create_density_matrix,
    mle_fit_density_matrix,
)


def _born_probabilities(rho: NDArray) -> dict[str, NDArray]:
    """Generate ideal outcome probabilities from independent Pauli eigenvectors."""
    eigenvectors = {
        "X": (np.array([1, 1]), np.array([1, -1])),
        "Y": (np.array([1, 1j]), np.array([1, -1j])),
        "Z": (np.array([1, 0]), np.array([0, 1])),
    }
    n_qubits = round(np.log2(rho.shape[0]))
    probabilities = {}
    for axes in product("XYZ", repeat=n_qubits):
        outcomes = []
        for bits in product((0, 1), repeat=n_qubits):
            vectors = [
                eigenvectors[axis][bit] / np.linalg.norm(eigenvectors[axis][bit])
                for axis, bit in zip(axes, bits, strict=True)
            ]
            vector = reduce(np.kron, vectors)
            outcomes.append(float(np.vdot(vector, rho @ vector).real))
        probabilities["".join(axes)] = np.asarray(outcomes)
    return probabilities


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive_y", "negative_y"])
def test_mle_preserves_signed_y_expectation(sign: float) -> None:
    """MLE reconstructs the supplied Y sign rather than conjugating the state."""
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    probabilities = {
        "X": np.array([0.5, 0.5]),
        "Y": np.array([(1 + sign) / 2, (1 - sign) / 2]),
        "Z": np.array([0.5, 0.5]),
    }
    linear = create_density_matrix(probabilities, mle_fit=False)
    reconstructed = create_density_matrix(probabilities, mle_fit=True)

    linear_y = float(np.trace(pauli_y @ linear).real)
    mle_y = float(np.trace(pauli_y @ reconstructed).real)
    assert linear_y == pytest.approx(sign, abs=1e-12)
    assert mle_y == pytest.approx(sign, abs=1e-4)
    np.testing.assert_allclose(reconstructed, linear, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([np.sqrt(0.37), np.exp(0.73j) * np.sqrt(0.63)]),
        np.array([np.sqrt(0.43), 0, 0, np.exp(1.13j) * np.sqrt(0.57)]),
        np.kron(np.array([1, 1j]), np.array([1, -1j])) / 2,
    ],
    ids=["complex_qubit", "complex_bell", "opposite_y_product"],
)
def test_tomography_preserves_complex_coherences(vector: NDArray) -> None:
    """Public linear and MLE reconstruction agree on independently generated Born data."""
    dim = vector.size
    expected = 0.8 * np.outer(vector, vector.conj()) + 0.2 * np.eye(dim) / dim
    probabilities = _born_probabilities(expected)

    linear = create_density_matrix(probabilities, mle_fit=False)
    reconstructed = create_density_matrix(probabilities, mle_fit=True)

    np.testing.assert_allclose(linear, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-4, rtol=0)
    assert np.min(np.linalg.eigvalsh(reconstructed)) >= -1e-10
    assert np.trace(reconstructed).real == pytest.approx(1.0, abs=1e-10)


def test_mle_preserves_a_general_mixed_two_qubit_state() -> None:
    """MLE retains real and imaginary coherences across every entry of a mixed state."""
    rng = np.random.default_rng(9735)
    factor = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    expected = factor @ factor.conj().T
    expected /= np.trace(expected)
    probabilities = _born_probabilities(expected)

    reconstructed = mle_fit_density_matrix(calculate_expected_values(probabilities))

    np.testing.assert_allclose(reconstructed, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("axis", ["X", "Z"])
def test_mle_keeps_real_pauli_states_unchanged(axis: str) -> None:
    """The coherence-sign correction leaves real-axis tomography unchanged."""
    probabilities = {label: np.array([0.5, 0.5]) for label in "XYZ"}
    probabilities[axis] = np.array([0.8, 0.2])
    linear = create_density_matrix(probabilities, mle_fit=False)

    reconstructed = create_density_matrix(probabilities, mle_fit=True)

    np.testing.assert_allclose(reconstructed, linear, atol=1e-4, rtol=0)
