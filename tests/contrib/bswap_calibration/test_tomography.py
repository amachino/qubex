"""Offline regression tests for the manual bSWAP tomography helpers."""

import numpy as np
import pytest
from qxpulse import Blank, FlatTop
from scipy.linalg import expm

from qubex.contrib.experiment.bswap_calibration.tomography import (
    BASES,
    PAULI,
    SQRT_BASES,
    density_from_counts,
    fit_local_phases,
    local_z,
    make_tomography_sequence,
    raw_unitary,
    sqrt_score,
    state_vector,
)


def counts_for(rho, bases):
    """Return ideal joint Pauli counts for a density matrix."""
    return np.array(
        [
            [
                np.trace(
                    rho
                    @ np.kron(
                        (PAULI["I"] + a * PAULI[b[0]]) / 2,
                        (PAULI["I"] + c * PAULI[b[1]]) / 2,
                    )
                ).real
                * 100000
                for a, c in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            ]
            for b in bases
        ]
    )


@pytest.mark.parametrize("state", [("0", "+i"), ("+i", "+"), ("1", "0")])
def test_linear_tomography_y_sign(state):
    """Linear tomography should preserve the prepared Y sign."""
    v = state_vector(state)
    rho = np.outer(v, v.conj())
    np.testing.assert_allclose(
        density_from_counts(counts_for(rho, BASES)), rho, atol=1e-12
    )


def test_sqrt_score_distinguishes_coherence_from_half_population():
    """The root score should distinguish coherence from an incoherent half population."""
    v = raw_unitary("sqrt_bswap", 0, 0.5, -0.3, 0.8) @ state_vector(("0", "0"))
    pure, _ = sqrt_score(counts_for(np.outer(v, v.conj()), SQRT_BASES))
    mixed, _ = sqrt_score(counts_for(np.diag([0.5, 0, 0, 0.5]), SQRT_BASES))
    np.testing.assert_allclose([pure, mixed], [1, 0.5], atol=1e-12)


@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
def test_local_phase_fit_retains_zz(kind):
    """Local phase calibration should retain the nonlocal ZZ angle."""
    states = [("0", "+"), ("1", "+"), ("+", "0"), ("+", "1")]
    raw = raw_unitary(kind, 0 if kind == "bswap" else 0.63, 0.41, -0.72, 0.37)
    rhos = np.array(
        [
            np.outer(raw @ state_vector(s), (raw @ state_vector(s)).conj())
            for s in states
        ]
    )
    fitted = fit_local_phases(kind, states, 0.83 * rhos + 0.17 * np.eye(4)[None] / 4)
    corrected = local_z(fitted["post_vz_rad"]) @ raw @ local_z(fitted["pre_vz_rad"])
    target = raw_unitary(kind, zeta=fitted["zz_phase_rad"])
    np.testing.assert_allclose(
        abs(np.trace(target.conj().T @ corrected)) / 4, 1, atol=1e-8
    )
    np.testing.assert_allclose(fitted["zz_phase_rad"], 0.37, atol=1e-6)


@pytest.mark.parametrize("kind", ["bswap", "sqrt_bswap"])
def test_flattened_preparation_and_post_vz(kind):
    """Flattened preparation and analysis should apply the requested VZ corrections."""
    qubits = ["A", "P"]
    x90 = {
        q: FlatTop(duration=24, tau=0, amplitude=np.pi / 48, sampling_period=2)
        for q in qubits
    }
    xpi = {
        q: FlatTop(duration=40, tau=0, amplitude=np.pi / 40, sampling_period=2)
        for q in qubits
    }

    def evolve(samples):
        """Evolve one sampled two-level quadrature waveform."""
        u = np.eye(2, dtype=complex)
        for value in samples:
            u = expm(-1j * (value.real * PAULI["X"] + value.imag * PAULI["Y"])) @ u
        return u

    pre = 0 if kind == "bswap" else 0.63
    raw = raw_unitary(kind, pre, 0.41, -0.72, 0.37)
    for state in [("0", "0"), ("1", "1"), ("+", "+"), ("+i", "+")]:
        counts = []
        for basis in BASES:
            schedule = make_tomography_sequence(
                qubits,
                "B",
                "S",
                Blank(32),
                x90,
                xpi,
                state,
                basis,
                pre_vz=[-pre, 0],
                post_vz=[-0.41, 0.72],
            )
            arrays = schedule.get_sampled_sequences()
            preparation = np.kron(*[evolve(arrays[q][:20]) for q in qubits])
            analysis = np.kron(*[evolve(arrays[q][36:]) for q in qubits])
            output = analysis @ raw @ preparation @ np.array([1, 0, 0, 0])
            counts.append(abs(output) ** 2 * 100000)
        target = raw_unitary(kind, zeta=0.37) @ state_vector(state)
        np.testing.assert_allclose(
            density_from_counts(np.array(counts)),
            np.outer(target, target.conj()),
            atol=1e-12,
        )
