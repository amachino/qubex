"""Small offline tomography and local-phase helpers for manual raw bSWAP calibration."""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qxpulse import Blank, PulseSchedule, VirtualZ, Waveform
from scipy.optimize import least_squares

BASES = tuple(a + b for a in "XYZ" for b in "XYZ")
SQRT_BASES = ("ZZ", "XX", "YY", "XY", "YX")
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag(np.array([1, -1], dtype=complex)),
}
VECTORS = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "+i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
}


def state_vector(state: Sequence[str]) -> NDArray[np.complex128]:
    """Return the two-qubit product vector for labels 0, 1, + or +i."""
    return np.asarray(
        np.kron(VECTORS[state[0]], VECTORS[state[1]]), dtype=np.complex128
    )


def local_z(angles: Sequence[float] | NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return local Z rotations for two angles in radians."""
    return np.asarray(
        np.kron(*[np.diag(np.exp(-0.5j * a * np.array([1, -1]))) for a in angles]),
        dtype=np.complex128,
    )


def raw_unitary(
    kind: str,
    pre_active: float = 0.0,
    post_active: float = 0.0,
    post_passive: float = 0.0,
    zeta: float = 0.0,
) -> NDArray[np.complex128]:
    """Use plus-i exchange and RZZ(zeta)=exp(-i*zeta*ZZ/2), in active/passive order."""
    if kind not in ("bswap", "sqrt_bswap"):
        raise ValueError("kind must be bswap or sqrt_bswap")
    theta = np.pi if kind == "bswap" else np.pi / 2
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    exchange = np.array(
        [[c, 0, 0, 1j * s], [0, 1, 0, 0], [0, 0, 1, 0], [1j * s, 0, 0, c]]
    )
    zz = np.diag(np.exp(-0.5j * zeta * np.array([1, -1, -1, 1])))
    return (
        local_z([post_active, post_passive]) @ exchange @ zz @ local_z([pre_active, 0])
    )


def make_tomography_sequence(
    qubits: Sequence[str],
    drive_label: str,
    sizzle_label: str,
    gate: Waveform,
    x90: Mapping[str, Waveform],
    xpi: Mapping[str, Waveform],
    state: Sequence[str],
    basis: str,
    pre_vz: Sequence[float] | NDArray[np.float64] = (0.0, 0.0),
    post_vz: Sequence[float] | NDArray[np.float64] = (0.0, 0.0),
    drive_on: bool = True,
) -> PulseSchedule:
    """
    Apply Pre Z through preparation axes and Post Z through analysis axes.

    Parameters
    ----------
    qubits : sequence of str
        Two qubit labels in active/passive order.
    gate : Waveform
        Complete gate pulse, including both ramps.
    state : sequence of str
        Prepared product-state labels from 0, 1, + and +i.
    basis : str
        Two analysis axes, each X, Y or Z.
    pre_vz, post_vz : sequence of float
        Two logical correction angles in radians.

    Returns
    -------
    PulseSchedule
        Preparation, fixed-start gate and analysis schedule. Padding precedes
        preparation; this function does not measure or connect to hardware.
    """
    prep_ns = max(
        [p.duration for p in x90.values()] + [p.duration for p in xpi.values()]
    )
    analysis_ns = max(p.duration for p in x90.values())
    with PulseSchedule([*qubits, drive_label, sizzle_label]) as sequence:
        for qi, (q, symbol) in enumerate(zip(qubits, state, strict=True)):
            if symbol in ("+", "+i"):
                axis = np.pi / 2 if symbol == "+" else np.pi
                pulse = x90[q].shifted(axis + pre_vz[qi])
            else:
                pulse = xpi[q] if symbol == "1" else Blank(prep_ns)
            if pulse.duration < prep_ns:
                sequence.add(q, Blank(prep_ns - pulse.duration))
            sequence.add(q, pulse)
        sequence.barrier()
        sequence.add(drive_label, gate if drive_on else Blank(gate.duration))
        sequence.add(sizzle_label, Blank(gate.duration))
        for q in qubits:
            sequence.add(q, Blank(gate.duration))
        sequence.barrier()
        for qi, (q, axis) in enumerate(zip(qubits, basis, strict=True)):
            sequence.add(q, VirtualZ(float(post_vz[qi])))
            pulse = (
                Blank(analysis_ns)
                if axis == "Z"
                else x90[q].shifted(-np.pi / 2 if axis == "X" else 0.0)
            )
            sequence.add(q, pulse)
            if pulse.duration < analysis_ns:
                sequence.add(q, Blank(analysis_ns - pulse.duration))
        sequence.barrier()
    return sequence


def density_from_counts(
    counts: ArrayLike, bases: Sequence[str] = BASES
) -> NDArray[np.complex128]:
    """
    Reconstruct a density matrix without clipping, projection or postselection.

    Parameters
    ----------
    counts : array_like
        Shape (9, 4) in the supplied Pauli-basis order. Signed, normalized
        estimates are accepted so combined-SPAM diagnostics remain unclipped.
    bases : sequence of str, default BASES
        All nine two-qubit Pauli measurement bases, exactly once.

    Returns
    -------
    ndarray
        Linear 4-by-4 estimate; positivity is not imposed.
    """
    counts = np.asarray(counts, dtype=float)
    if (
        counts.shape != (9, 4)
        or set(bases) != set(BASES)
        or not np.isfinite(counts).all()
        or np.any(counts.sum(axis=-1) <= 0)
    ):
        raise ValueError("Need all nine Pauli bases with finite, nonempty shot counts")
    probs = counts / counts.sum(axis=-1, keepdims=True)
    values = {"II": 1.0}
    for axis in "XYZ":
        values[axis + "I"] = float(
            np.mean(
                [
                    p @ np.array([1, 1, -1, -1])
                    for p, b in zip(probs, bases, strict=True)
                    if b[0] == axis
                ]
            )
        )
        values["I" + axis] = float(
            np.mean(
                [
                    p @ np.array([1, -1, 1, -1])
                    for p, b in zip(probs, bases, strict=True)
                    if b[1] == axis
                ]
            )
        )
    for p, b in zip(probs, bases, strict=True):
        values[b] = float(p @ np.array([1, -1, -1, 1]))
    return np.asarray(
        sum(
            values[a + b] * np.kron(PAULI[a], PAULI[b])
            for a, b in itertools.product("IXYZ", repeat=2)
        )
        / 4,
        dtype=np.complex128,
    )


def sqrt_score(counts: ArrayLike) -> tuple[float, float]:
    """Phase-agnostic even-Bell score and approximate shot variance, not gate fidelity."""
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (5, 4):
        raise ValueError("Need counts in ZZ,XX,YY,XY,YX order")
    if not np.isfinite(counts).all() or np.any(counts.sum(axis=-1) <= 0):
        return float("nan"), float("nan")
    shots = counts.sum(axis=-1)
    probs = counts / shots[:, None]
    even = float(probs[0, 0] + probs[0, 3])
    corr = probs[1:] @ np.array([1, -1, -1, 1])
    u, v = corr[0] - corr[1], corr[2] + corr[3]
    radius = np.hypot(u, v)
    variances = np.maximum(0, 1 - corr**2) / shots[1:]
    var_c = (
        (u * u) * (variances[0] + variances[1])
        + (v * v) * (variances[2] + variances[3])
    ) / (16 * max(radius * radius, 1e-12))
    if radius < 1e-6:
        var_c = float(variances.sum() / 16)
    return float(even / 2 + radius / 4), float(
        max(1e-12, even * (1 - even) / (4 * shots[0]) + var_c)
    )


def fit_local_phases(
    kind: str, states: Sequence[Sequence[str]], density_matrices: ArrayLike
) -> dict[str, Any]:
    """
    Fit local phases and residual ZZ with passive Pre fixed to zero.

    Parameters
    ----------
    kind : str
        `bswap` or `sqrt_bswap`, selecting the fixed ideal exchange angle.
    states : sequence of two-label sequences
        Independently prepared product-state inputs in active/passive order.
    density_matrices : array_like
        Corresponding linear density estimates, shape (input, 4, 4).

    Returns
    -------
    dict
        Measured phase angles and inverse VZ corrections in radians, plus
        coherence residuals. Residual ZZ is retained, not corrected locally.
    """
    observed = np.asarray(density_matrices, dtype=complex)
    mask = ~np.eye(4, dtype=bool)
    initial = [state_vector(s) for s in states]

    def predicted(p: NDArray[np.float64]) -> NDArray[np.complex128]:
        pre, pa, pp, zz = (0.0, *p) if kind == "bswap" else p
        unitary = raw_unitary(kind, pre, pa, pp, zz)
        vectors = [unitary @ v for v in initial]
        return np.array([np.outer(v, v.conj()) for v in vectors])

    npar = 3 if kind == "bswap" else 4
    reference = predicted(np.zeros(npar))
    strengths = np.linalg.norm(observed[:, mask], axis=1) / np.linalg.norm(
        reference[:, mask], axis=1
    )
    if np.min(strengths) < 0.1 or not np.isfinite(strengths).all():
        raise ValueError("Insufficient measured coherence for a phase calibration")

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        diff = observed[:, mask] - strengths[:, None] * predicted(p)[:, mask]
        return np.r_[diff.real.ravel(), diff.imag.ravel()]

    rng = np.random.default_rng(42)
    lower = np.array([-2 * np.pi] * (npar - 1) + [-np.pi / 2])
    upper = -lower
    fits = [
        least_squares(residual, p, bounds=(lower, upper), max_nfev=2000)
        for p in [
            np.zeros(npar),
            *[rng.uniform(lower * 0.99, upper * 0.99) for _ in range(16)],
        ]
    ]
    successful = [f for f in fits if f.success]
    if not successful:
        raise ValueError("Local phase model did not converge")
    fit = min(successful, key=lambda f: np.sum(f.fun**2))
    pre, pa, pp, zz = (0.0, *fit.x) if kind == "bswap" else fit.x
    wrap = lambda a: float(np.angle(np.exp(1j * a)))
    return dict(
        gate_kind=kind,
        pre_active_rad=wrap(pre),
        post_active_rad=wrap(pa),
        post_passive_rad=wrap(pp),
        zz_phase_rad=float(zz),
        pre_vz_rad=[wrap(-pre), 0.0],
        post_vz_rad=[wrap(-pa), wrap(-pp)],
        coherence_scales=strengths.tolist(),
        coherence_residual_rms=float(np.sqrt(np.mean(fit.fun**2))),
        scope="fixed-placement local phase calibration; residual ZZ retained; not gate fidelity",
    )
