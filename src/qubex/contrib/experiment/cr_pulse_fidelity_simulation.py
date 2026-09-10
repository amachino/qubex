"""Simulate dissipative fidelity limits of a ZX90 cross-resonance gate."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from qubex.pulse import PulseSchedule

_SimulationMode = Literal[
    "ideal",
    "idle_coherence",
    "cr_on_coherence",
    "cr_on_dissipative",
]

_QUTRIT_DIMENSION = 3
_FULL_DIMENSION = _QUTRIT_DIMENSION**2
_COMPUTATIONAL_DIMENSION = 4
_IDENTITY_QUTRIT = np.eye(_QUTRIT_DIMENSION, dtype=np.complex128)
_IDENTITY_FULL = np.eye(_FULL_DIMENSION, dtype=np.complex128)
_X_GE = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    dtype=np.complex128,
)
_Z_GE = np.diag([1.0, -1.0, 0.0]).astype(np.complex128)
_CONTROL_X = np.kron(_X_GE, _IDENTITY_QUTRIT)
_TARGET_X = np.kron(_IDENTITY_QUTRIT, _X_GE)
_CONTROL_Z = np.kron(_Z_GE, _IDENTITY_QUTRIT)
_TARGET_Z = np.kron(_IDENTITY_QUTRIT, _Z_GE)
_ZX = np.kron(_Z_GE, _X_GE)
_SUPEROPERATOR_DIMENSION = _FULL_DIMENSION**2
_CONTROL_DIAGONAL_VECTOR_INDICES = np.asarray(
    [
        control * _QUTRIT_DIMENSION
        + target_row
        + (control * _QUTRIT_DIMENSION + target_column) * _FULL_DIMENSION
        for control in range(_QUTRIT_DIMENSION)
        for target_column in range(_QUTRIT_DIMENSION)
        for target_row in range(_QUTRIT_DIMENSION)
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class IdleQubitNoise:
    """Describe idle coherence times in ns."""

    t1: float
    t2_echo: float

    def __post_init__(self) -> None:
        """Validate positive finite or infinite coherence times."""
        _validate_lifetime(self.t1, name="t1")
        _validate_lifetime(self.t2_echo, name="t2_echo")
        if not np.isinf(self.t1) and (
            np.isinf(self.t2_echo) or self.t2_echo > 2 * self.t1
        ):
            warnings.warn(
                "t2_echo exceeds 2 * t1; idle pure dephasing will be clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class CrOnNoise:
    """
    Describe effective CR-active lifetimes and transition rates for one lobe.

    Lifetimes use ns and rates use inverse ns. Simulation APIs accept an
    optional separate noise model for the negative-amplitude lobe. By default,
    this instance is applied to both CR signs, assuming that dissipative rates
    are invariant under drive-phase reversal. The target fields describe one
    effective model; the simulator does not explicitly condition them on the
    control state.
    """

    gamma_control_g_to_e: float
    gamma_control_e_to_g: float
    gamma_control_e_to_f: float
    gamma_control_f_to_e: float
    target_t1rho: float
    target_leakage_rate: float
    target_seepage_rate: float
    gamma_phi_control: float = 0.0
    gamma_phi_rho_target: float = 0.0

    def __post_init__(self) -> None:
        """Validate nonnegative rates and a positive target T1rho."""
        for name in (
            "gamma_control_g_to_e",
            "gamma_control_e_to_g",
            "gamma_control_e_to_f",
            "gamma_control_f_to_e",
            "target_leakage_rate",
            "target_seepage_rate",
            "gamma_phi_control",
            "gamma_phi_rho_target",
        ):
            _validate_nonnegative_finite(getattr(self, name), name=name)
        _validate_lifetime(self.target_t1rho, name="target_t1rho")


@dataclass(frozen=True)
class ZX90GateTiming:
    """Describe the semantic timing of an ideal echoed or un-echoed ZX90."""

    cr_lobe_duration: float
    echo: bool
    control_pi_duration: float | None = None
    echo_margin_duration: float = 0.0

    def __post_init__(self) -> None:
        """Validate the semantic gate timing."""
        _validate_positive_finite(self.cr_lobe_duration, name="cr_lobe_duration")
        if not isinstance(self.echo, bool):
            raise TypeError("echo must be a boolean.")
        if self.echo and self.control_pi_duration is None:
            raise ValueError(
                "control_pi_duration is required when constructing echoed timing."
            )
        if self.control_pi_duration is not None:
            if self.echo:
                _validate_positive_finite(
                    self.control_pi_duration,
                    name="control_pi_duration",
                )
            else:
                _validate_nonnegative_finite(
                    self.control_pi_duration,
                    name="control_pi_duration",
                )
        _validate_nonnegative_finite(
            self.echo_margin_duration,
            name="echo_margin_duration",
        )
        if not self.echo and (
            (self.control_pi_duration is not None and self.control_pi_duration > 0)
            or self.echo_margin_duration > 0
        ):
            raise ValueError(
                "An un-echoed ZX90 cannot contain an echo pi pulse or margin."
            )

    @property
    def duration(self) -> float:
        """Return the total semantic gate duration in ns."""
        if not self.echo:
            return self.cr_lobe_duration
        control_pi_duration = self.control_pi_duration
        if control_pi_duration is None:  # pragma: no cover - dataclass invariant
            raise RuntimeError("Echoed timing has no control pi duration.")
        return 2 * (
            self.cr_lobe_duration + control_pi_duration + 2 * self.echo_margin_duration
        )


@dataclass(frozen=True)
class CrPulseFidelitySimulationResult:
    """Store the three fidelity limits, average leakage, and model metadata."""

    idle_coherence_limited_fidelity: float
    cr_on_coherence_limited_fidelity: float
    cr_on_dissipative_limited_fidelity: float
    average_leakage: float
    model_metadata: dict[str, object]


@dataclass(frozen=True)
class CrTargetDecayPrediction:
    """Store A/B control populations and target observables at each n value."""

    control_populations: NDArray[np.float64]
    target_x: NDArray[np.float64]
    target_f: NDArray[np.float64]


def _validate_positive_finite(value: float, *, name: str) -> None:
    """Validate a positive finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number.")


def _validate_nonnegative_finite(value: float, *, name: str) -> None:
    """Validate a nonnegative finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a nonnegative finite real number.")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite real number.")


def _validate_lifetime(value: float, *, name: str) -> None:
    """Validate a positive lifetime, permitting positive infinity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    if np.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be positive.")


def extract_zx90_gate_timing(gate: PulseSchedule) -> ZX90GateTiming:
    """
    Extract semantic CR and echo timing from a Qubex ZX90 gate object.

    Parameters
    ----------
    gate
        ZX90 schedule exposing Qubex cross-resonance timing metadata.

    Returns
    -------
    ZX90GateTiming
        CR-lobe, echo-pulse, and echo-margin durations in ns.

    Raises
    ------
    TypeError
        If the gate's echo marker is not boolean.
    ValueError
        If required metadata are absent or inconsistent with the schedule.

    Notes
    -----
    A supported gate exposes the `cr_duration` and `echo` metadata used by
    Qubex `CrossResonance` objects. Echoed gates must also expose their
    physical control `pi_pulse`. Any remaining duration surrounding each pi
    pulse is assigned equally to the two CR-off margins.
    """
    try:
        cr_duration = float(vars(gate)["cr_duration"])
        echo_value = vars(gate)["echo"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The ZX90 gate must expose `cr_duration` and `echo` metadata."
        ) from exc
    if not isinstance(echo_value, bool):
        raise TypeError("The ZX90 gate `echo` metadata must be a boolean.")
    echo = echo_value
    if not echo:
        if not np.isclose(gate.duration, cr_duration, rtol=0.0, atol=1e-9):
            raise ValueError(
                "The un-echoed ZX90 contains timing that cannot be assigned to "
                "CR-active or CR-off intervals."
            )
        return ZX90GateTiming(cr_lobe_duration=cr_duration, echo=False)

    pi_pulse = vars(gate).get("pi_pulse")
    if pi_pulse is None:
        raise ValueError("An echoed ZX90 must expose its control `pi_pulse`.")
    try:
        pi_duration = float(pi_pulse.duration)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("The echoed ZX90 pi pulse must expose its duration.") from exc
    echo_layer_duration = (float(gate.duration) - 2 * cr_duration) / 2
    margin_duration = (echo_layer_duration - pi_duration) / 2
    if margin_duration < -1e-9:
        raise ValueError("The echoed ZX90 timing is shorter than its component pulses.")
    return ZX90GateTiming(
        cr_lobe_duration=cr_duration,
        echo=True,
        control_pi_duration=pi_duration,
        echo_margin_duration=max(margin_duration, 0.0),
    )


def _transition(row: int, column: int) -> NDArray[np.complex128]:
    """Return a local qutrit transition operator |row><column|."""
    operator = np.zeros((_QUTRIT_DIMENSION,) * 2, dtype=np.complex128)
    operator[row, column] = 1.0
    return operator


def _control(operator: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Embed an operator on the control qutrit."""
    return np.kron(operator, _IDENTITY_QUTRIT)


def _target(operator: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Embed an operator on the target qutrit."""
    return np.kron(_IDENTITY_QUTRIT, operator)


def _collapse(
    rate: float,
    operator: NDArray[np.complex128],
) -> NDArray[np.complex128] | None:
    """Return a scaled collapse operator, omitting a zero-rate channel."""
    return None if rate == 0 else np.sqrt(rate) * operator


def _idle_rates(noise: IdleQubitNoise) -> tuple[float, float, bool]:
    """Return idle relaxation/dephasing rates and whether dephasing was clamped."""
    gamma_1 = 0.0 if np.isinf(noise.t1) else 1 / noise.t1
    inverse_t2 = 0.0 if np.isinf(noise.t2_echo) else 1 / noise.t2_echo
    raw_gamma_phi = inverse_t2 - gamma_1 / 2
    return gamma_1, max(0.0, raw_gamma_phi), raw_gamma_phi < 0


def _idle_collapse_operators(
    control_noise: IdleQubitNoise,
    target_noise: IdleQubitNoise,
) -> tuple[NDArray[np.complex128], ...]:
    """Build idle amplitude-damping and pure-dephasing operators."""
    operators: list[NDArray[np.complex128] | None] = []
    for noise, embed in (
        (control_noise, _control),
        (target_noise, _target),
    ):
        gamma_1, gamma_phi, _ = _idle_rates(noise)
        operators.extend(
            (
                _collapse(gamma_1, embed(_transition(0, 1))),
                _collapse(gamma_phi / 2, embed(_Z_GE)),
            )
        )
    return tuple(operator for operator in operators if operator is not None)


def _cr_collapse_operators(
    noise: CrOnNoise,
    *,
    include_leakage: bool,
) -> tuple[NDArray[np.complex128], ...]:
    """Build the CR-active collapse operators of the effective qutrit model."""
    operators: list[NDArray[np.complex128] | None] = [
        _collapse(noise.gamma_control_g_to_e, _control(_transition(1, 0))),
        _collapse(noise.gamma_control_e_to_g, _control(_transition(0, 1))),
        _collapse(noise.gamma_phi_control / 2, _CONTROL_Z),
    ]
    if include_leakage:
        operators.extend(
            (
                _collapse(noise.gamma_control_e_to_f, _control(_transition(2, 1))),
                _collapse(noise.gamma_control_f_to_e, _control(_transition(1, 2))),
            )
        )

    plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    minus = np.array([1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    dressed_rate = 0.0 if np.isinf(noise.target_t1rho) else 1 / (2 * noise.target_t1rho)
    operators.extend(
        (
            _collapse(dressed_rate, _target(np.outer(minus, plus.conj()))),
            _collapse(dressed_rate, _target(np.outer(plus, minus.conj()))),
            _collapse(noise.gamma_phi_rho_target / 2, _TARGET_X),
        )
    )
    if include_leakage:
        operators.extend(
            (
                _collapse(noise.target_leakage_rate, _target(_transition(2, 0))),
                _collapse(noise.target_leakage_rate, _target(_transition(2, 1))),
                _collapse(noise.target_seepage_rate / 2, _target(_transition(0, 2))),
                _collapse(noise.target_seepage_rate / 2, _target(_transition(1, 2))),
            )
        )
    return tuple(operator for operator in operators if operator is not None)


def _liouvillian(
    hamiltonian: NDArray[np.complex128],
    collapse_operators: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    """Build a column-vectorized Lindblad generator."""
    generator = -1j * (
        np.kron(_IDENTITY_FULL, hamiltonian) - np.kron(hamiltonian.T, _IDENTITY_FULL)
    )
    for operator in collapse_operators:
        product = operator.conj().T @ operator
        generator += np.kron(operator.conj(), operator)
        generator -= 0.5 * np.kron(_IDENTITY_FULL, product)
        generator -= 0.5 * np.kron(product.T, _IDENTITY_FULL)
    return generator


def _segment_map(
    hamiltonian: NDArray[np.complex128],
    duration: float,
    collapse_operators: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    """Return the finite-duration superoperator for one segment."""
    if duration == 0:
        return np.eye(_SUPEROPERATOR_DIMENSION, dtype=np.complex128)
    return np.asarray(
        expm(_liouvillian(hamiltonian, collapse_operators) * duration),
        dtype=np.complex128,
    )


def _unitary_map(unitary: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the column-vectorized superoperator of a unitary."""
    return np.kron(unitary.conj(), unitary)


def _compose(
    *maps: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Compose chronologically ordered superoperators."""
    total = np.eye(_SUPEROPERATOR_DIMENSION, dtype=np.complex128)
    for superoperator in maps:
        total = superoperator @ total
    return total


def _x_layer_map(
    duration: float,
    control_angle: float,
    target_angle: float,
    collapse_operators: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    """Return an ideal simultaneous X layer evolving under idle noise."""
    if duration == 0:
        unitary = np.asarray(
            expm(-0.5j * (control_angle * _CONTROL_X + target_angle * _TARGET_X)),
            dtype=np.complex128,
        )
        return _unitary_map(unitary)
    hamiltonian = (control_angle * _CONTROL_X + target_angle * _TARGET_X) / (
        2 * duration
    )
    return _segment_map(hamiltonian, duration, collapse_operators)


def _parallel_x_layer_map(
    control_duration: float,
    control_angle: float,
    target_duration: float,
    target_angle: float,
    collapse_operators: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    """Return simultaneous X rotations without stretching the shorter pulse."""
    instantaneous_control_angle = control_angle if control_duration == 0 else 0.0
    instantaneous_target_angle = target_angle if target_duration == 0 else 0.0
    maps: list[NDArray[np.complex128]] = []
    if instantaneous_control_angle != 0 or instantaneous_target_angle != 0:
        instantaneous = np.asarray(
            expm(
                -0.5j
                * (
                    instantaneous_control_angle * _CONTROL_X
                    + instantaneous_target_angle * _TARGET_X
                )
            ),
            dtype=np.complex128,
        )
        maps.append(_unitary_map(instantaneous))

    segment_start = 0.0
    for segment_end in sorted(
        {duration for duration in (control_duration, target_duration) if duration > 0}
    ):
        segment_duration = segment_end - segment_start
        hamiltonian = np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128)
        if segment_start < control_duration:
            hamiltonian += control_angle * _CONTROL_X / (2 * control_duration)
        if segment_start < target_duration:
            hamiltonian += target_angle * _TARGET_X / (2 * target_duration)
        maps.append(_segment_map(hamiltonian, segment_duration, collapse_operators))
        segment_start = segment_end
    return _compose(*maps)


def _control_diagonal_generator(
    generator: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Restrict a two-qutrit generator to control-diagonal density blocks."""
    return generator[
        np.ix_(
            _CONTROL_DIAGONAL_VECTOR_INDICES,
            _CONTROL_DIAGONAL_VECTOR_INDICES,
        )
    ]


def _conditioned_target_generator(
    control_state: int,
    target_operators: tuple[NDArray[np.complex128], ...],
) -> NDArray[np.complex128]:
    """Build one unit-rate target dissipator conditioned on a control state."""
    projector = _transition(control_state, control_state)
    operators = tuple(np.kron(projector, operator) for operator in target_operators)
    zero_hamiltonian = np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128)
    return _control_diagonal_generator(_liouvillian(zero_hamiltonian, operators))


@dataclass(frozen=True)
class CrTargetDecayModel:
    """Cache the control-transition-aware physical model for A/B target decay."""

    zx90_duration: float
    control_population_generator: NDArray[np.float64]
    control_population_map: NDArray[np.float64]
    base_generator: NDArray[np.complex128]
    t1rho_generators: tuple[NDArray[np.complex128], NDArray[np.complex128]]
    leakage_generators: tuple[NDArray[np.complex128], NDArray[np.complex128]]
    seepage_generators: tuple[NDArray[np.complex128], NDArray[np.complex128]]

    def gate_map(
        self,
        gamma_1rho: tuple[float, float],
        leakage_rates: tuple[float, float],
        seepage_rates: tuple[float, float],
    ) -> NDArray[np.complex128]:
        """Build one full un-echoed ZX90 map for state-conditioned target rates."""
        rates = (*gamma_1rho, *leakage_rates, *seepage_rates)
        for index, rate in enumerate(rates):
            _validate_nonnegative_finite(rate, name=f"target_rate[{index}]")
        generator = self.base_generator.copy()
        for rate, component in zip(
            gamma_1rho,
            self.t1rho_generators,
            strict=True,
        ):
            generator += rate * component
        for rate, component in zip(
            leakage_rates,
            self.leakage_generators,
            strict=True,
        ):
            generator += rate * component
        for rate, component in zip(
            seepage_rates,
            self.seepage_generators,
            strict=True,
        ):
            generator += rate * component
        return np.asarray(expm(generator * self.zx90_duration), dtype=np.complex128)

    def predict_pair(
        self,
        n_values: NDArray[np.int64],
        initial_control_populations: tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        initial_target_x: tuple[float, float],
        initial_target_f: tuple[float, float],
        gamma_1rho: tuple[float, float],
        leakage_rates: tuple[float, float],
        seepage_rates: tuple[float, float],
    ) -> tuple[CrTargetDecayPrediction, CrTargetDecayPrediction]:
        """Predict the A/B observables while sharing one finite-duration map."""
        counts = np.asarray(n_values)
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("n_values must be a one-dimensional integer array.")
        counts = np.asarray(counts, dtype=np.int64)
        if np.any(counts < 0) or np.any(np.diff(counts) < 0):
            raise ValueError("n_values must be nonnegative and nondecreasing.")
        rates = (*gamma_1rho, *leakage_rates, *seepage_rates)
        for index, rate in enumerate(rates):
            _validate_nonnegative_finite(rate, name=f"target_rate[{index}]")
        gamma_by_control = np.asarray(
            (*gamma_1rho, 0.5 * sum(gamma_1rho)),
            dtype=np.float64,
        )
        leakage_by_control = np.asarray(
            (*leakage_rates, 0.5 * sum(leakage_rates)),
            dtype=np.float64,
        )
        seepage_by_control = np.asarray(
            (*seepage_rates, 0.5 * sum(seepage_rates)),
            dtype=np.float64,
        )
        control_generator = self.control_population_generator
        target_x_generator = control_generator - np.diag(
            gamma_by_control + leakage_by_control
        )
        target_f_generator = np.block(
            [
                [control_generator, np.zeros((3, 3))],
                [
                    np.diag(leakage_by_control),
                    control_generator
                    - np.diag(leakage_by_control + seepage_by_control),
                ],
            ]
        )
        x_map = np.asarray(
            expm(target_x_generator * self.zx90_duration),
            dtype=np.float64,
        )
        f_map = np.asarray(
            expm(target_f_generator * self.zx90_duration),
            dtype=np.float64,
        )
        return (
            self._predict_observables(
                x_map,
                f_map,
                4 * counts,
                initial_control_populations[0],
                initial_target_x[0],
                initial_target_f[0],
            ),
            self._predict_observables(
                x_map,
                f_map,
                4 * counts,
                initial_control_populations[1],
                initial_target_x[1],
                initial_target_f[1],
            ),
        )

    def _predict_observables(
        self,
        x_map: NDArray[np.float64],
        f_map: NDArray[np.float64],
        repetition_counts: NDArray[np.int64],
        initial_control_population: NDArray[np.float64],
        initial_target_x: float,
        initial_target_f: float,
    ) -> CrTargetDecayPrediction:
        """Propagate the exact closed observable subspace of the Lindblad model."""
        control = np.asarray(initial_control_population, dtype=np.float64)
        if (
            control.shape != (_QUTRIT_DIMENSION,)
            or not np.all(np.isfinite(control))
            or np.any(control < 0)
            or not np.isclose(np.sum(control), 1.0)
        ):
            raise ValueError("initial_control_population must be a probability vector.")
        if not -1.0 <= initial_target_x <= 1.0:
            raise ValueError("initial_target_x must lie in [-1, 1].")
        if not 0.0 <= initial_target_f < 1.0:
            raise ValueError("initial_target_f must lie in [0, 1).")
        x_state = control * (1 - initial_target_f) * initial_target_x
        f_state = np.concatenate((control, control * initial_target_f))
        control_populations = np.empty((repetition_counts.size, 3))
        target_x = np.empty(repetition_counts.size)
        target_f = np.empty(repetition_counts.size)
        previous_count = 0
        for index, count_value in enumerate(repetition_counts):
            count = int(count_value)
            for _ in range(count - previous_count):
                control = self.control_population_map @ control
                x_state = x_map @ x_state
                f_state = f_map @ f_state
            f_population = float(np.sum(f_state[3:]))
            survival = 1 - f_population
            if survival <= np.finfo(float).eps:
                raise ValueError("Target computational survival reached zero.")
            control_populations[index] = control
            target_f[index] = f_population
            target_x[index] = float(np.sum(x_state) / survival)
            previous_count = count
        return CrTargetDecayPrediction(control_populations, target_x, target_f)

    def _predict_one(
        self,
        gate_map: NDArray[np.complex128],
        repetition_counts: NDArray[np.int64],
        initial_control_population: NDArray[np.float64],
        initial_target_x: float,
        initial_target_f: float,
    ) -> CrTargetDecayPrediction:
        """Propagate one control preparation through a shared ZX90 map."""
        control_population = np.asarray(
            initial_control_population,
            dtype=np.float64,
        )
        if (
            control_population.shape != (_QUTRIT_DIMENSION,)
            or not np.all(np.isfinite(control_population))
            or np.any(control_population < 0)
            or not np.isclose(np.sum(control_population), 1.0)
        ):
            raise ValueError("initial_control_population must be a probability vector.")
        if not -1.0 <= initial_target_x <= 1.0:
            raise ValueError("initial_target_x must lie in [-1, 1].")
        if not 0.0 <= initial_target_f < 1.0:
            raise ValueError("initial_target_f must lie in [0, 1).")

        target_density = np.zeros(
            (_QUTRIT_DIMENSION, _QUTRIT_DIMENSION),
            dtype=np.complex128,
        )
        computational_population = 1 - initial_target_f
        target_density[:2, :2] = (
            0.5
            * computational_population
            * np.array(
                [[1.0, initial_target_x], [initial_target_x, 1.0]],
                dtype=np.complex128,
            )
        )
        target_density[2, 2] = initial_target_f
        target_vector = target_density.reshape(-1, order="F")
        state = np.concatenate(
            [population * target_vector for population in control_population]
        )

        n_points = repetition_counts.size
        control_populations = np.empty(
            (n_points, _QUTRIT_DIMENSION),
            dtype=np.float64,
        )
        target_x = np.empty(n_points, dtype=np.float64)
        target_f = np.empty(n_points, dtype=np.float64)
        previous_count = 0
        for point_index, count_value in enumerate(repetition_counts):
            count = int(count_value)
            for _ in range(count - previous_count):
                state = gate_map @ state
            block_vectors = state.reshape((_QUTRIT_DIMENSION, -1))
            target_state = np.sum(block_vectors, axis=0).reshape(
                (_QUTRIT_DIMENSION,) * 2,
                order="F",
            )
            control_populations[point_index] = np.real(
                [
                    np.trace(
                        block_vector.reshape(
                            (_QUTRIT_DIMENSION,) * 2,
                            order="F",
                        )
                    )
                    for block_vector in block_vectors
                ]
            )
            target_f[point_index] = float(np.real(target_state[2, 2]))
            computational_survival = 1 - target_f[point_index]
            if computational_survival <= np.finfo(float).eps:
                raise ValueError("Target computational survival reached zero.")
            target_x[point_index] = float(
                np.real(np.trace(_X_GE @ target_state)) / computational_survival
            )
            previous_count = count
        return CrTargetDecayPrediction(
            control_populations=control_populations,
            target_x=target_x,
            target_f=target_f,
        )


def prepare_cr_target_decay_model(
    zx90_duration: float,
    gamma_control_g_to_e: float,
    gamma_control_e_to_g: float,
    gamma_control_e_to_f: float,
    gamma_control_f_to_e: float,
) -> CrTargetDecayModel:
    """
    Prepare the control-transition-aware physical predictor for A/B.

    The full un-echoed ZX90 is modeled as a single `pi/2` ZX segment. Target
    rotating-frame relaxation and leakage/seepage are conditioned on the
    instantaneous control state. The unmeasured control-F target rate is the
    arithmetic mean of the corresponding control-G and control-E rates.

    Parameters
    ----------
    zx90_duration
        Duration in ns of one full un-echoed ZX90.
    gamma_control_g_to_e, gamma_control_e_to_g
        Control GE transition rates in `1/ns`.
    gamma_control_e_to_f, gamma_control_f_to_e
        Control EF transition rates in `1/ns`.

    Returns
    -------
    CrTargetDecayModel
        Reusable model that predicts target X/F and control populations for
        the two control preparations.

    Notes
    -----
    The model retains the full 27-dimensional control-diagonal Lindblad
    generator for validation. Repeated fitting propagates its exact closed
    observable subspace instead: three control populations, three
    control-conditioned target-X contributions, and six joint control/target-F
    populations. This avoids repeated 27-by-27 exponentials without changing
    the predicted A/B observables.
    """
    _validate_positive_finite(zx90_duration, name="zx90_duration")
    control_rates = (
        gamma_control_g_to_e,
        gamma_control_e_to_g,
        gamma_control_e_to_f,
        gamma_control_f_to_e,
    )
    for name, rate in zip(
        (
            "gamma_control_g_to_e",
            "gamma_control_e_to_g",
            "gamma_control_e_to_f",
            "gamma_control_f_to_e",
        ),
        control_rates,
        strict=True,
    ):
        _validate_nonnegative_finite(rate, name=name)

    control_operators = tuple(
        operator
        for operator in (
            _collapse(gamma_control_g_to_e, _control(_transition(1, 0))),
            _collapse(gamma_control_e_to_g, _control(_transition(0, 1))),
            _collapse(gamma_control_e_to_f, _control(_transition(2, 1))),
            _collapse(gamma_control_f_to_e, _control(_transition(1, 2))),
        )
        if operator is not None
    )
    hamiltonian = np.pi * _ZX / (4 * zx90_duration)
    base_generator = _control_diagonal_generator(
        _liouvillian(hamiltonian, control_operators)
    )

    plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    minus = np.array([1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
    dressed_operators = (
        np.outer(minus, plus.conj()) / np.sqrt(2),
        np.outer(plus, minus.conj()) / np.sqrt(2),
    )
    leakage_operators = (_transition(2, 0), _transition(2, 1))
    seepage_operators = (
        _transition(0, 2) / np.sqrt(2),
        _transition(1, 2) / np.sqrt(2),
    )

    def averaged_control_f_generators(
        target_operators: tuple[NDArray[np.complex128], ...],
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        generators = tuple(
            _conditioned_target_generator(control_state, target_operators)
            for control_state in range(_QUTRIT_DIMENSION)
        )
        return (
            generators[0] + 0.5 * generators[2],
            generators[1] + 0.5 * generators[2],
        )

    control_population_generator = np.array(
        [
            [-gamma_control_g_to_e, gamma_control_e_to_g, 0.0],
            [
                gamma_control_g_to_e,
                -(gamma_control_e_to_g + gamma_control_e_to_f),
                gamma_control_f_to_e,
            ],
            [0.0, gamma_control_e_to_f, -gamma_control_f_to_e],
        ],
        dtype=np.float64,
    )
    return CrTargetDecayModel(
        zx90_duration=float(zx90_duration),
        control_population_generator=control_population_generator,
        control_population_map=np.asarray(
            expm(control_population_generator * zx90_duration),
            dtype=np.float64,
        ),
        base_generator=base_generator,
        t1rho_generators=averaged_control_f_generators(dressed_operators),
        leakage_generators=averaged_control_f_generators(leakage_operators),
        seepage_generators=averaged_control_f_generators(seepage_operators),
    )


@dataclass(frozen=True)
class CrEchoDecayModel:
    """Cache the physical model used to predict CR echo-decay protocols."""

    cr_lobe_duration: float
    positive_generator: NDArray[np.complex128]
    negative_generator: NDArray[np.complex128]
    control_dephasing_generator: NDArray[np.complex128]
    target_dephasing_generator: NDArray[np.complex128]
    echo_layer: NDArray[np.complex128]
    control_x: NDArray[np.complex128]
    simultaneous_x: NDArray[np.complex128]
    target_z: NDArray[np.complex128]

    def gate_map(
        self,
        gamma_phi_control: float,
        gamma_phi_rho_target: float,
    ) -> NDArray[np.complex128]:
        """Build one ZX90 map with the specified CR-on dephasing rates."""
        _validate_nonnegative_finite(
            gamma_phi_control,
            name="gamma_phi_control",
        )
        _validate_nonnegative_finite(
            gamma_phi_rho_target,
            name="gamma_phi_rho_target",
        )
        variable_generator = (
            gamma_phi_control * self.control_dephasing_generator
            + gamma_phi_rho_target * self.target_dephasing_generator
        )
        positive_cr = np.asarray(
            expm(
                (self.positive_generator + variable_generator) * self.cr_lobe_duration
            ),
            dtype=np.complex128,
        )
        negative_cr = np.asarray(
            expm(
                (self.negative_generator + variable_generator) * self.cr_lobe_duration
            ),
            dtype=np.complex128,
        )
        return _compose(
            positive_cr,
            self.echo_layer,
            negative_cr,
            self.echo_layer,
        )

    def predict_control_x(
        self,
        n_values: NDArray[np.int64],
        gamma_phi_control: float,
    ) -> NDArray[np.float64]:
        """Predict control X while varying only control CR-on dephasing."""
        gate_map = self.gate_map(gamma_phi_control, 0.0)
        control_block = _compose(
            gate_map,
            self.control_x,
            gate_map,
            self.simultaneous_x,
            gate_map,
            self.control_x,
            gate_map,
        )
        plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
        return _expectation_curve(
            control_block,
            n_values,
            _state_vector(np.kron(plus, plus)),
            _CONTROL_X,
        )

    def predict_target_z(
        self,
        n_values: NDArray[np.int64],
        gamma_phi_rho_target: float,
    ) -> NDArray[np.float64]:
        """Predict target Z while varying only target CR-on dephasing."""
        gate_map = self.gate_map(0.0, gamma_phi_rho_target)
        target_block = _compose(gate_map, self.target_z, gate_map)
        ground = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        return _expectation_curve(
            target_block,
            2 * n_values,
            _state_vector(np.kron(ground, ground)),
            _TARGET_Z,
        )


def prepare_cr_echo_decay_model(
    timing: ZX90GateTiming,
    control_idle: IdleQubitNoise,
    target_idle: IdleQubitNoise,
    cr_noise: CrOnNoise,
    control_x180_duration: float,
    target_x180_duration: float,
    *,
    cr_noise_negative: CrOnNoise | None = None,
) -> CrEchoDecayModel:
    """
    Prepare a reusable physical predictor for the CR echo-decay curves.

    Parameters
    ----------
    timing
        Semantic timing of an echoed ZX90 gate in ns.
    control_idle, target_idle
        Idle-noise models used during CR-off intervals.
    cr_noise
        Effective noise model for the positive-amplitude CR lobe.
    control_x180_duration, target_x180_duration
        Physical X180 durations in ns for the control-echo protocol.
    cr_noise_negative
        Optional separate noise model for the negative-amplitude CR lobe.

    Returns
    -------
    CrEchoDecayModel
        Cached generators and maps for the control-T2-echo and
        target-T2rho-echo protocols.

    Notes
    -----
    Every map and generator independent of the variable CR-on dephasing rate
    is precomputed here. `cr_noise` applies to the positive CR lobe;
    `cr_noise_negative` applies to the negative lobe and defaults to the
    positive model. The fitted dephasing parameter remains common to both
    signs because the current echo protocols do not distinguish them.
    Measurement fitting is deliberately left to
    `cr_pulse_coherence_analysis`.
    """
    if not timing.echo:
        raise ValueError("Echo-decay forward modeling requires an echoed ZX90 gate.")
    control_pi_duration = timing.control_pi_duration
    if control_pi_duration is None:  # pragma: no cover - dataclass invariant
        raise ValueError("Echoed ZX90 timing requires control_pi_duration.")
    _validate_nonnegative_finite(
        control_x180_duration,
        name="control_x180_duration",
    )
    _validate_nonnegative_finite(
        target_x180_duration,
        name="target_x180_duration",
    )
    idle_operators = _idle_collapse_operators(control_idle, target_idle)
    margin = _segment_map(
        np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128),
        timing.echo_margin_duration,
        idle_operators,
    )
    control_pi = _x_layer_map(
        control_pi_duration,
        np.pi,
        0.0,
        idle_operators,
    )
    echo_layer = _compose(margin, control_pi, margin)

    negative_noise = cr_noise if cr_noise_negative is None else cr_noise_negative
    positive_base_noise = replace(
        cr_noise,
        gamma_phi_control=0.0,
        gamma_phi_rho_target=0.0,
    )
    negative_base_noise = replace(
        negative_noise,
        gamma_phi_control=0.0,
        gamma_phi_rho_target=0.0,
    )
    positive_operators = _cr_collapse_operators(
        positive_base_noise,
        include_leakage=True,
    )
    negative_operators = _cr_collapse_operators(
        negative_base_noise,
        include_leakage=True,
    )
    positive_hamiltonian = np.pi * _ZX / (8 * timing.cr_lobe_duration)
    zero_hamiltonian = np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128)
    control_dephasing_operator = _CONTROL_Z / np.sqrt(2)
    target_dephasing_operator = _TARGET_X / np.sqrt(2)
    target_z_unitary = np.asarray(
        expm(-0.5j * np.pi * _TARGET_Z),
        dtype=np.complex128,
    )
    return CrEchoDecayModel(
        cr_lobe_duration=timing.cr_lobe_duration,
        positive_generator=_liouvillian(positive_hamiltonian, positive_operators),
        negative_generator=_liouvillian(-positive_hamiltonian, negative_operators),
        control_dephasing_generator=_liouvillian(
            zero_hamiltonian,
            (control_dephasing_operator,),
        ),
        target_dephasing_generator=_liouvillian(
            zero_hamiltonian,
            (target_dephasing_operator,),
        ),
        echo_layer=echo_layer,
        control_x=_x_layer_map(
            control_x180_duration,
            np.pi,
            0.0,
            idle_operators,
        ),
        simultaneous_x=_parallel_x_layer_map(
            control_x180_duration,
            np.pi,
            target_x180_duration,
            np.pi,
            idle_operators,
        ),
        target_z=_unitary_map(target_z_unitary),
    )


def _zx90_map(
    timing: ZX90GateTiming,
    control_idle: IdleQubitNoise,
    target_idle: IdleQubitNoise,
    cr_noise: CrOnNoise,
    mode: _SimulationMode,
    cr_noise_negative: CrOnNoise | None = None,
) -> NDArray[np.complex128]:
    """Build a semantic ZX90 map with CR-on/off noise switching."""
    idle_operators = (
        () if mode == "ideal" else _idle_collapse_operators(control_idle, target_idle)
    )
    negative_noise = cr_noise if cr_noise_negative is None else cr_noise_negative
    if mode in ("ideal", "idle_coherence"):
        positive_operators = idle_operators
    else:
        include_leakage = mode == "cr_on_dissipative"
        positive_operators = _cr_collapse_operators(
            cr_noise,
            include_leakage=include_leakage,
        )

    lobe_angle = np.pi / 4 if timing.echo else np.pi / 2
    positive_cr = _segment_map(
        lobe_angle * _ZX / (2 * timing.cr_lobe_duration),
        timing.cr_lobe_duration,
        positive_operators,
    )
    if not timing.echo:
        return positive_cr
    negative_operators = (
        idle_operators
        if mode in ("ideal", "idle_coherence")
        else _cr_collapse_operators(
            negative_noise,
            include_leakage=mode == "cr_on_dissipative",
        )
    )
    negative_cr = _segment_map(
        -lobe_angle * _ZX / (2 * timing.cr_lobe_duration),
        timing.cr_lobe_duration,
        negative_operators,
    )
    margin = _segment_map(
        np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128),
        timing.echo_margin_duration,
        idle_operators,
    )
    control_pi_duration = timing.control_pi_duration
    if control_pi_duration is None:  # pragma: no cover - dataclass invariant
        raise ValueError("Echoed ZX90 timing requires control_pi_duration.")
    control_pi = _x_layer_map(
        control_pi_duration,
        np.pi,
        0.0,
        idle_operators,
    )
    return _compose(
        positive_cr,
        margin,
        control_pi,
        margin,
        negative_cr,
        margin,
        control_pi,
        margin,
    )


def _computational_map(
    full_map: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Restrict a full two-qutrit map to computational inputs and outputs."""
    full_indices = (0, 1, 3, 4)
    restricted = np.zeros(
        (_COMPUTATIONAL_DIMENSION**2,) * 2,
        dtype=np.complex128,
    )
    operator_indices = [(row, col) for col in full_indices for row in full_indices]
    for input_column, (input_row, input_col) in enumerate(operator_indices):
        full_input_column = input_row + input_col * _FULL_DIMENSION
        for output_column, (output_row, output_col) in enumerate(operator_indices):
            full_output_column = output_row + output_col * _FULL_DIMENSION
            restricted[output_column, input_column] = full_map[
                full_output_column,
                full_input_column,
            ]
    return restricted


def _physical_probability(value: float, *, name: str) -> float:
    """Clip roundoff at probability boundaries and reject material violations."""
    tolerance = 1e-10
    if value < -tolerance or value > 1 + tolerance or not np.isfinite(value):
        raise RuntimeError(f"{name}={value} lies outside the physical interval [0, 1].")
    return float(np.clip(value, 0.0, 1.0))


def _fidelity_and_survival(
    noisy_map: NDArray[np.complex128],
    ideal_map: NDArray[np.complex128],
) -> tuple[float, float]:
    """Return projected average fidelity and computational survival."""
    relative_map = ideal_map.conj().T @ noisy_map
    computational_map = _computational_map(relative_map)
    entanglement_fidelity = float(
        np.real(np.trace(computational_map)) / _COMPUTATIONAL_DIMENSION**2
    )
    mixed_state = np.zeros(_COMPUTATIONAL_DIMENSION**2, dtype=np.complex128)
    for index in range(_COMPUTATIONAL_DIMENSION):
        mixed_state[index + index * _COMPUTATIONAL_DIMENSION] = (
            1 / _COMPUTATIONAL_DIMENSION
        )
    output = computational_map @ mixed_state
    survival = float(
        np.real(
            sum(
                output[index + index * _COMPUTATIONAL_DIMENSION]
                for index in range(_COMPUTATIONAL_DIMENSION)
            )
        )
    )
    average_fidelity = (_COMPUTATIONAL_DIMENSION * entanglement_fidelity + survival) / (
        _COMPUTATIONAL_DIMENSION + 1
    )
    return (
        _physical_probability(average_fidelity, name="average_fidelity"),
        _physical_probability(survival, name="average_survival"),
    )


def simulate_cr_pulse_fidelity(
    gate: PulseSchedule | ZX90GateTiming,
    control_idle: IdleQubitNoise,
    target_idle: IdleQubitNoise,
    cr_noise: CrOnNoise,
    *,
    cr_noise_negative: CrOnNoise | None = None,
) -> CrPulseFidelitySimulationResult:
    """
    Calculate three dissipative ZX90 fidelity limits and average leakage.

    Parameters
    ----------
    gate
        Echoed or un-echoed ZX90 schedule, or its semantic timing.
    control_idle, target_idle
        Idle-noise models used during CR-off intervals.
    cr_noise
        Effective noise model for the positive-amplitude CR lobe.
    cr_noise_negative
        Optional noise model for the negative-amplitude lobe. Defaults to
        `cr_noise`.

    Returns
    -------
    CrPulseFidelitySimulationResult
        Idle, CR-on coherence, and CR-on dissipative fidelity limits together
        with average leakage and model assumptions.

    Notes
    -----
    The two qutrits are propagated with piecewise Lindblad superoperators.
    CR-active intervals use either idle noise or the supplied CR-on models;
    `cr_noise` applies to the positive lobe and `cr_noise_negative` to the
    negative lobe. By default both signs use `cr_noise`, which is a
    reasonable first approximation when rates depend primarily on CR power.
    Echo pulses and margins use idle noise. A noise-free simulation of the
    same semantic gate is removed before the channel is projected onto the
    four-dimensional computational subspace.

    All durations and lifetimes use ns and all rates use `1/ns`. The
    dissipative fidelity uses the trace-decreasing formula
    `(4 * entanglement_fidelity + average_survival) / 5` so leakage is
    counted as infidelity.
    """
    timing = (
        gate if isinstance(gate, ZX90GateTiming) else extract_zx90_gate_timing(gate)
    )
    negative_noise = cr_noise if cr_noise_negative is None else cr_noise_negative
    ideal_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "ideal",
        negative_noise,
    )
    idle_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "idle_coherence",
        negative_noise,
    )
    coherence_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "cr_on_coherence",
        negative_noise,
    )
    dissipative_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "cr_on_dissipative",
        negative_noise,
    )
    idle_fidelity, _ = _fidelity_and_survival(idle_map, ideal_map)
    coherence_fidelity, _ = _fidelity_and_survival(
        coherence_map,
        ideal_map,
    )
    dissipative_fidelity, dissipative_survival = _fidelity_and_survival(
        dissipative_map,
        ideal_map,
    )
    _, _, control_idle_clamped = _idle_rates(control_idle)
    _, _, target_idle_clamped = _idle_rates(target_idle)
    if not timing.echo:
        cr_lobe_rate_model = "positive_lobe_only"
    elif negative_noise == cr_noise:
        cr_lobe_rate_model = "same_for_positive_and_negative_drive_signs"
    else:
        cr_lobe_rate_model = "independent_positive_and_negative_drive_signs"
    return CrPulseFidelitySimulationResult(
        idle_coherence_limited_fidelity=idle_fidelity,
        cr_on_coherence_limited_fidelity=coherence_fidelity,
        cr_on_dissipative_limited_fidelity=dissipative_fidelity,
        average_leakage=1 - dissipative_survival,
        model_metadata={
            "time_unit": "ns",
            "rate_unit": "1/ns",
            "t2_star_used": False,
            "coherent_errors_included": False,
            "coherent_leakage_included": False,
            "target_polarization_asymptote": 0.0,
            "target_leakage_outward_model": "equal_incoherent_from_g_and_e",
            "target_seepage_return_model": "equal_incoherent_to_g_and_e",
            "cr_lobe_rate_model": cr_lobe_rate_model,
            "cr_on_rate_target_state_dependence": (
                "effective_average_not_explicitly_modeled"
            ),
            "reference_operation": "noise_off_same_semantic_gate",
            "gate_schedule_interpretation": "semantic_cr_echo_timing",
            "zx_hamiltonian_qutrit_extension": (
                "Z_control=diag(1,-1,0), X_target couples only g-e"
            ),
            "control_qutrit_dephasing_extension": "diag(1,-1,0)",
            "idle_pure_dephasing_clamped": {
                "control": control_idle_clamped,
                "target": target_idle_clamped,
            },
        },
    )


def _expectation(
    state_vector: NDArray[np.complex128],
    observable: NDArray[np.complex128],
) -> float:
    """Evaluate an observable from a vectorized density matrix."""
    density_matrix = state_vector.reshape(
        (_FULL_DIMENSION,) * 2,
        order="F",
    )
    return float(np.real(np.trace(observable @ density_matrix)))


def _state_vector(state: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the vectorized pure-state density matrix."""
    return np.outer(state, state.conj()).reshape(-1, order="F")


def _expectation_curve(
    block: NDArray[np.complex128],
    repetition_counts: NDArray[np.int64],
    initial_state: NDArray[np.complex128],
    observable: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Propagate one state incrementally over increasing repetition counts."""
    if repetition_counts.ndim != 1 or not np.issubdtype(
        repetition_counts.dtype, np.integer
    ):
        raise ValueError("repetition counts must be a one-dimensional integer array.")
    if np.any(repetition_counts < 0) or np.any(np.diff(repetition_counts) < 0):
        raise ValueError("repetition counts must be nonnegative and nondecreasing.")
    values = np.empty(repetition_counts.shape, dtype=np.float64)
    state = initial_state
    previous_count = 0
    for index, count_value in enumerate(repetition_counts):
        count = int(count_value)
        for _ in range(count - previous_count):
            state = block @ state
        values[index] = _expectation(state, observable)
        previous_count = count
    return values


__all__ = [
    "CrEchoDecayModel",
    "CrOnNoise",
    "CrPulseFidelitySimulationResult",
    "CrTargetDecayModel",
    "CrTargetDecayPrediction",
    "IdleQubitNoise",
    "ZX90GateTiming",
    "extract_zx90_gate_timing",
    "prepare_cr_echo_decay_model",
    "prepare_cr_target_decay_model",
    "simulate_cr_pulse_fidelity",
]
