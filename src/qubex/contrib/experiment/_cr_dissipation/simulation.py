"""
Simulate semantic two-qutrit CR evolution and fidelity limits.

The module defines the qutrit operator basis, constructs cached piecewise
Lindblad channels, evaluates repeated protocol observables, and optionally
propagates fitted-rate uncertainty into leakage-aware average fidelity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from .types import IdleNoiseParameters

_COMPLEX = np.complex128
_DIMENSION = 9
_SUPER_DIMENSION = _DIMENSION**2
_MAX_SEGMENT_CACHE_ENTRIES = 256


def _basis(index: int) -> NDArray[np.complex128]:
    """Return one vector from the local three-level computational basis."""
    vector = np.zeros(3, dtype=_COMPLEX)
    vector[index] = 1.0
    return vector


G = _basis(0)
E = _basis(1)
F = _basis(2)
I3 = np.eye(3, dtype=_COMPLEX)
X_GE = np.outer(G, E.conj()) + np.outer(E, G.conj())
Y_GE = -1j * np.outer(G, E.conj()) + 1j * np.outer(E, G.conj())
Z_GE = np.outer(G, G.conj()) - np.outer(E, E.conj())


def tensor(
    left: NDArray[np.complex128], right: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Return a two-qutrit tensor product."""
    return np.asarray(np.kron(left, right), dtype=_COMPLEX)


X_CONTROL = tensor(X_GE, I3)
Y_CONTROL = tensor(Y_GE, I3)
Z_CONTROL = tensor(Z_GE, I3)
X_TARGET = tensor(I3, X_GE)
Y_TARGET = tensor(I3, Y_GE)
Z_TARGET = tensor(I3, Z_GE)
ZX = tensor(Z_GE, X_GE)
IDENTITY = np.eye(_DIMENSION, dtype=_COMPLEX)


# Semantic operations and dissipative-rate inputs


@dataclass(frozen=True)
class SemanticSegment:
    """Represent one constant-Hamiltonian segment in ns."""

    duration_ns: float
    hamiltonian: NDArray[np.complex128]
    cr_active: bool
    label: str


@dataclass(frozen=True)
class SemanticUnitary:
    """Represent one zero-duration exact frame operation."""

    unitary: NDArray[np.complex128]
    label: str


SemanticOperation = SemanticSegment | SemanticUnitary
SuperoperatorCache = MutableMapping[tuple[object, ...], NDArray[np.complex128]]


@dataclass(frozen=True)
class CrNoiseRates:
    """Store primitive CR-active rates in 1/ns."""

    control_e_to_g: float = 0.0
    control_g_to_e: float = 0.0
    control_e_to_f: float = 0.0
    control_f_to_e: float = 0.0
    control_pure_dephasing: float = 0.0
    target_t1rho: float = 0.0
    target_leakage: float = 0.0
    target_seepage: float = 0.0
    target_rotating_frame_pure_dephasing: float = 0.0


def state_density(control: str, target: str) -> NDArray[np.complex128]:
    """Return a product-state density matrix for semantic protocol inputs."""
    states = {
        "g": G,
        "e": E,
        "+x": (G + E) / np.sqrt(2.0),
        "+y": (G + 1j * E) / np.sqrt(2.0),
    }
    try:
        vector = np.kron(states[control.lower()], states[target.lower()])
    except KeyError as exc:
        raise ValueError(f"Unsupported semantic state {exc.args[0]!r}.") from exc
    return np.asarray(np.outer(vector, vector.conj()), dtype=_COMPLEX)


def rotation_unitary(
    operator: NDArray[np.complex128], angle_rad: float
) -> NDArray[np.complex128]:
    """Return `exp(-i angle operator / 2)`."""
    return np.asarray(expm(-0.5j * float(angle_rad) * operator), dtype=_COMPLEX)


def rotation_hamiltonian(
    operator: NDArray[np.complex128],
    angle_rad: float,
    duration_ns: float,
) -> NDArray[np.complex128]:
    """Return a constant semantic Hamiltonian for one intended rotation."""
    if not np.isfinite(duration_ns) or duration_ns <= 0.0:
        raise ValueError("A physical rotation duration must be positive and finite.")
    return np.asarray(float(angle_rad) * operator / (2.0 * duration_ns), dtype=_COMPLEX)


def simultaneous_rotation_segments(
    rotations: Sequence[tuple[NDArray[np.complex128], float, float, str]],
    *,
    cr_active: bool = False,
) -> tuple[SemanticSegment, ...]:
    """Split simultaneous rotations without stretching shorter pulses."""
    if not rotations:
        return ()
    durations = sorted({float(item[2]) for item in rotations})
    if durations[0] <= 0.0 or not np.all(np.isfinite(durations)):
        raise ValueError("Simultaneous rotation durations must be positive and finite.")
    segments: list[SemanticSegment] = []
    start = 0.0
    for end in durations:
        hamiltonian = np.zeros((_DIMENSION, _DIMENSION), dtype=_COMPLEX)
        labels = []
        for operator, angle, duration, label in rotations:
            if float(duration) > start:
                hamiltonian += rotation_hamiltonian(operator, angle, float(duration))
                labels.append(label)
        segments.append(
            SemanticSegment(
                duration_ns=end - start,
                hamiltonian=hamiltonian,
                cr_active=cr_active,
                label="+".join(labels),
            )
        )
        start = end
    return tuple(segments)


@lru_cache(maxsize=64)
def _cached_hamiltonian_liouvillian(
    hamiltonian_bytes: bytes,
) -> NDArray[np.complex128]:
    """Build a commutator Liouvillian from a canonical matrix encoding."""
    hamiltonian = np.frombuffer(hamiltonian_bytes, dtype=_COMPLEX).reshape(
        _DIMENSION, _DIMENSION
    )
    return np.asarray(
        -1j * (np.kron(IDENTITY, hamiltonian) - np.kron(hamiltonian.T, IDENTITY)),
        dtype=_COMPLEX,
    )


def hamiltonian_liouvillian(
    hamiltonian: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return a cached column-vectorized commutator Liouvillian."""
    canonical = np.ascontiguousarray(hamiltonian, dtype=_COMPLEX)
    if canonical.shape != (_DIMENSION, _DIMENSION):
        raise ValueError("Hamiltonian must act on the two-qutrit Hilbert space.")
    return _cached_hamiltonian_liouvillian(canonical.tobytes())


def dissipator_superoperator(
    operator: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return the unit-coefficient Lindblad dissipator superoperator."""
    operator = np.asarray(operator, dtype=_COMPLEX)
    if operator.shape != (_DIMENSION, _DIMENSION):
        raise ValueError("Dissipator operator must act on two qutrits.")
    product = operator.conj().T @ operator
    return (
        np.kron(operator.conj(), operator)
        - 0.5 * np.kron(IDENTITY, product)
        - 0.5 * np.kron(product.T, IDENTITY)
    )


def unitary_superoperator(unitary: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the column-vectorized superoperator for a unitary."""
    unitary = np.asarray(unitary, dtype=_COMPLEX)
    if unitary.shape != (_DIMENSION, _DIMENSION):
        raise ValueError("Unitary must act on the two-qutrit Hilbert space.")
    return np.asarray(np.kron(unitary.conj(), unitary), dtype=_COMPLEX)


def _transition(
    destination: NDArray[np.complex128], source: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Return a local transition operator from source to destination."""
    return np.outer(destination, source.conj())


def _embedded_transition(
    role: Literal["control", "target"],
    destination: NDArray[np.complex128],
    source: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Embed one local transition on the selected qutrit."""
    local = _transition(destination, source)
    return tensor(local, I3) if role == "control" else tensor(I3, local)


@lru_cache(maxsize=1)
def _dissipator_components() -> Mapping[str, NDArray[np.complex128]]:
    """Build the rate-independent Lindblad components once per process."""
    plus_x = (G + E) / np.sqrt(2.0)
    minus_x = (G - E) / np.sqrt(2.0)
    operators = {
        "control_e_to_g": _embedded_transition("control", G, E),
        "control_g_to_e": _embedded_transition("control", E, G),
        "control_e_to_f": _embedded_transition("control", F, E),
        "control_f_to_e": _embedded_transition("control", E, F),
        "control_phi": Z_CONTROL,
        "target_e_to_g": _embedded_transition("target", G, E),
        "target_x_plus_to_minus": _embedded_transition("target", minus_x, plus_x),
        "target_x_minus_to_plus": _embedded_transition("target", plus_x, minus_x),
        "target_phi": X_TARGET,
        "target_g_to_f": _embedded_transition("target", F, G),
        "target_e_to_f": _embedded_transition("target", F, E),
        "target_f_to_g": _embedded_transition("target", G, F),
        "target_f_to_e": _embedded_transition("target", E, F),
        "target_idle_phi": Z_TARGET,
    }
    return {
        name: np.asarray(dissipator_superoperator(operator), dtype=_COMPLEX)
        for name, operator in operators.items()
    }


def _weighted_liouvillian(
    terms: Sequence[tuple[float, str]],
    *,
    negative_components: frozenset[str] = frozenset(),
) -> NDArray[np.complex128]:
    """Combine rates with cached Lindblad components and validate their signs."""
    components = _dissipator_components()
    result = np.zeros((_SUPER_DIMENSION, _SUPER_DIMENSION), dtype=_COMPLEX)
    for coefficient, component_name in terms:
        if not np.isfinite(coefficient):
            raise ValueError("Dissipative coefficients must be finite.")
        if coefficient < 0.0 and component_name not in negative_components:
            raise ValueError("Nominal dissipative coefficients must be nonnegative.")
        if coefficient != 0.0:
            result += coefficient * components[component_name]
    return result


@lru_cache(maxsize=64)
def _idle_dissipative_liouvillian(
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
) -> NDArray[np.complex128]:
    """Return a cached idle dissipative Liouvillian."""
    return _weighted_liouvillian(
        (
            (control_idle.relaxation_rate_per_ns, "control_e_to_g"),
            (control_idle.pure_dephasing_rate_per_ns / 2.0, "control_phi"),
            (target_idle.relaxation_rate_per_ns, "target_e_to_g"),
            (target_idle.pure_dephasing_rate_per_ns / 2.0, "target_idle_phi"),
        ),
    )


@lru_cache(maxsize=128)
def _cr_dissipative_liouvillian(
    rates: CrNoiseRates,
    include_leakage: bool,
    signed_control_pure_dephasing: float | None,
    signed_target_pure_dephasing: float | None,
) -> NDArray[np.complex128]:
    """Return a cached CR-active dissipative Liouvillian."""
    control_phi = (
        rates.control_pure_dephasing
        if signed_control_pure_dephasing is None
        else float(signed_control_pure_dephasing)
    )
    target_phi = (
        rates.target_rotating_frame_pure_dephasing
        if signed_target_pure_dephasing is None
        else float(signed_target_pure_dephasing)
    )
    terms: list[tuple[float, str]] = [
        (rates.control_e_to_g, "control_e_to_g"),
        (rates.control_g_to_e, "control_g_to_e"),
        (control_phi / 2.0, "control_phi"),
        (rates.target_t1rho / 2.0, "target_x_plus_to_minus"),
        (rates.target_t1rho / 2.0, "target_x_minus_to_plus"),
        (target_phi / 2.0, "target_phi"),
    ]
    if include_leakage:
        terms.extend(
            [
                (rates.control_e_to_f, "control_e_to_f"),
                (rates.control_f_to_e, "control_f_to_e"),
                (rates.target_leakage, "target_g_to_f"),
                (rates.target_leakage, "target_e_to_f"),
                (rates.target_seepage / 2.0, "target_f_to_g"),
                (rates.target_seepage / 2.0, "target_f_to_e"),
            ]
        )
    negative_components = frozenset(
        name
        for name, signed_rate in (
            ("control_phi", signed_control_pure_dephasing),
            ("target_phi", signed_target_pure_dephasing),
        )
        if signed_rate is not None
    )
    return _weighted_liouvillian(terms, negative_components=negative_components)


def _segment_cache_key(
    segment: SemanticSegment,
    *,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    cr_rates: CrNoiseRates | None,
    include_leakage: bool,
    signed_control_pure_dephasing: float | None,
    signed_target_pure_dephasing: float | None,
) -> tuple[object, ...]:
    """Return a stable key for one segment superoperator."""
    noise_key: tuple[object, ...]
    if segment.cr_active and cr_rates is not None:
        noise_key = (
            "cr",
            cr_rates,
            include_leakage,
            signed_control_pure_dephasing,
            signed_target_pure_dephasing,
        )
    else:
        noise_key = ("idle", control_idle, target_idle)
    return (
        float(segment.duration_ns),
        bool(segment.cr_active),
        np.asarray(segment.hamiltonian, dtype=_COMPLEX).tobytes(),
        *noise_key,
    )


def segment_superoperator(
    segment: SemanticSegment,
    *,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    cr_rates: CrNoiseRates | None,
    include_leakage: bool,
    signed_control_pure_dephasing: float | None = None,
    signed_target_pure_dephasing: float | None = None,
    cache: SuperoperatorCache | None = None,
) -> NDArray[np.complex128]:
    """Return one piecewise semantic evolution superoperator."""
    key = _segment_cache_key(
        segment,
        control_idle=control_idle,
        target_idle=target_idle,
        cr_rates=cr_rates,
        include_leakage=include_leakage,
        signed_control_pure_dephasing=signed_control_pure_dephasing,
        signed_target_pure_dephasing=signed_target_pure_dephasing,
    )
    if cache is not None and key in cache:
        return cache[key]
    liouvillian = hamiltonian_liouvillian(segment.hamiltonian)
    dissipative = (
        _cr_dissipative_liouvillian(
            cr_rates,
            include_leakage,
            signed_control_pure_dephasing,
            signed_target_pure_dephasing,
        )
        if segment.cr_active and cr_rates is not None
        else _idle_dissipative_liouvillian(control_idle, target_idle)
    )
    result = np.asarray(
        expm((liouvillian + dissipative) * segment.duration_ns),
        dtype=_COMPLEX,
    )
    if cache is not None:
        if len(cache) >= _MAX_SEGMENT_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = result
    return result


def compose_channel(
    operations: Sequence[SemanticOperation],
    *,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    cr_rates: CrNoiseRates | None,
    include_leakage: bool,
    signed_control_pure_dephasing: float | None = None,
    signed_target_pure_dephasing: float | None = None,
    cache: SuperoperatorCache | None = None,
) -> NDArray[np.complex128]:
    """Compose a chronological semantic channel."""
    channel = np.eye(_SUPER_DIMENSION, dtype=_COMPLEX)
    for operation in operations:
        if isinstance(operation, SemanticUnitary):
            step = unitary_superoperator(operation.unitary)
        else:
            step = segment_superoperator(
                operation,
                control_idle=control_idle,
                target_idle=target_idle,
                cr_rates=cr_rates,
                include_leakage=include_leakage,
                signed_control_pure_dephasing=signed_control_pure_dephasing,
                signed_target_pure_dephasing=signed_target_pure_dephasing,
                cache=cache,
            )
        channel = step @ channel
    return channel


def apply_channel(
    channel: NDArray[np.complex128],
    density_matrix: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Apply a column-vectorized channel to one density matrix."""
    vector = np.asarray(density_matrix, dtype=_COMPLEX).reshape(-1, order="F")
    return np.asarray(
        (channel @ vector).reshape((_DIMENSION, _DIMENSION), order="F"), dtype=_COMPLEX
    )


def simulate_repeated_observable(
    block_operations: Sequence[SemanticOperation],
    repetition_counts: ArrayLike,
    initial_density: NDArray[np.complex128],
    observable: NDArray[np.complex128],
    *,
    control_idle: IdleNoiseParameters,
    target_idle: IdleNoiseParameters,
    cr_rates: CrNoiseRates | None,
    include_leakage: bool = True,
    signed_control_pure_dephasing: float | None = None,
    signed_target_pure_dephasing: float | None = None,
    cache: SuperoperatorCache | None = None,
) -> NDArray[np.float64]:
    """Simulate an expectation after integer powers of one protocol block."""
    block_channel = compose_channel(
        block_operations,
        control_idle=control_idle,
        target_idle=target_idle,
        cr_rates=cr_rates,
        include_leakage=include_leakage,
        signed_control_pure_dephasing=signed_control_pure_dephasing,
        signed_target_pure_dephasing=signed_target_pure_dephasing,
        cache=cache,
    )
    values = []
    counts = np.asarray(repetition_counts, dtype=np.int64).reshape(-1)
    for count in counts:
        state = apply_channel(
            np.linalg.matrix_power(block_channel, int(count)),
            initial_density,
        )
        values.append(float(np.real(np.trace(state @ observable))))
    return np.asarray(values, dtype=np.float64)


def noiseless_channel(
    operations: Sequence[SemanticOperation],
) -> NDArray[np.complex128]:
    """Return the exact semantic channel without any dissipation."""
    channel = np.eye(_SUPER_DIMENSION, dtype=_COMPLEX)
    for operation in operations:
        if isinstance(operation, SemanticUnitary):
            step = unitary_superoperator(operation.unitary)
        else:
            step = expm(
                hamiltonian_liouvillian(operation.hamiltonian) * operation.duration_ns
            )
        channel = step @ channel
    return np.asarray(channel, dtype=_COMPLEX)


def leakage_aware_average_fidelity(
    channel: NDArray[np.complex128],
    ideal_channel: NDArray[np.complex128],
) -> tuple[float, float, float]:
    """Return generalized average fidelity, entanglement fidelity, and survival."""
    channel = np.asarray(channel, dtype=_COMPLEX)
    ideal_channel = np.asarray(ideal_channel, dtype=_COMPLEX)
    if (
        channel.shape != (_SUPER_DIMENSION, _SUPER_DIMENSION)
        or ideal_channel.shape != channel.shape
    ):
        raise ValueError("Channels must be 81 by 81 two-qutrit superoperators.")
    relative = ideal_channel.conj().T @ channel
    computational_indices = (0, 1, 3, 4)
    dimension = len(computational_indices)
    entanglement_sum = 0.0j
    survival_sum = 0.0
    for row, physical_row in enumerate(computational_indices):
        for column, physical_column in enumerate(computational_indices):
            operator = np.zeros((_DIMENSION, _DIMENSION), dtype=_COMPLEX)
            operator[physical_row, physical_column] = 1.0
            output = apply_channel(relative, operator)
            entanglement_sum += output[physical_row, physical_column]
            if row == column:
                survival_sum += float(
                    np.real(
                        sum(output[index, index] for index in computational_indices)
                    )
                )
    entanglement_fidelity = float(np.real(entanglement_sum) / dimension**2)
    survival = float(survival_sum / dimension)
    average = (dimension * entanglement_fidelity + survival) / (dimension + 1)
    return (
        float(np.clip(average, 0.0, 1.0)),
        float(entanglement_fidelity),
        float(survival),
    )


def propagate_fidelity_uncertainty(
    central_rates: Mapping[str, float],
    covariance: NDArray[np.float64],
    evaluator: Callable[[Mapping[str, float]], float],
    *,
    positive_parameters: Sequence[str],
    psd_relative_tolerance: float = 1e-10,
) -> tuple[float | None, bool]:
    """Propagate conditional covariance with a scaled unscented transform."""
    names = tuple(central_rates)
    dimension = len(names)
    if covariance.shape != (dimension, dimension):
        raise ValueError(
            "Covariance shape must match the sigma-point parameter vector."
        )
    if dimension == 0:
        return 0.0, True
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, _ = np.linalg.eigh(covariance)
    scale = max(float(np.max(eigenvalues)), float(np.finfo(np.float64).eps))
    if float(np.min(eigenvalues)) < -psd_relative_tolerance * scale:
        return None, False
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive = set(positive_parameters)
    mean = np.asarray([float(central_rates[name]) for name in names], dtype=np.float64)
    transform = np.eye(dimension, dtype=np.float64)
    transformed_mean = mean.copy()
    for index, name in enumerate(names):
        if name in positive:
            if mean[index] <= 0.0 or not np.isfinite(mean[index]):
                raise ValueError("Log-space sigma-point parameters must be positive.")
            transformed_mean[index] = np.log(mean[index])
            transform[index, index] = 1.0 / mean[index]
    transformed_covariance = transform @ covariance @ transform.T
    transformed_covariance = 0.5 * (transformed_covariance + transformed_covariance.T)
    values, vectors = np.linalg.eigh(transformed_covariance)
    values = np.maximum(values, 0.0)
    square_root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    sigma_points = [transformed_mean]
    spread = np.sqrt(float(dimension))
    for column in range(dimension):
        offset = spread * square_root[:, column]
        sigma_points.extend((transformed_mean + offset, transformed_mean - offset))
    weights_mean = np.full(2 * dimension + 1, 1.0 / (2.0 * dimension))
    weights_mean[0] = 0.0
    weights_covariance = weights_mean.copy()
    weights_covariance[0] = 2.0
    evaluated = []
    for point in sigma_points:
        rate_point: dict[str, float] = {}
        for index, name in enumerate(names):
            rate_point[name] = (
                float(np.exp(point[index])) if name in positive else float(point[index])
            )
        evaluated.append(float(evaluator(rate_point)))
    evaluated_array = np.asarray(evaluated, dtype=np.float64)
    if not np.all(np.isfinite(evaluated_array)):
        return None, False
    sigma_mean = float(weights_mean @ evaluated_array)
    variance = float(weights_covariance @ np.square(evaluated_array - sigma_mean))
    return float(np.sqrt(max(variance, 0.0))), True
