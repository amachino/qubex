"""Simulate dissipative fidelity limits of a ZX90 cross-resonance gate."""

from __future__ import annotations

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


@dataclass(frozen=True)
class IdleQubitNoise:
    """Describe idle coherence times in ns."""

    t1: float
    t2_echo: float

    def __post_init__(self) -> None:
        """Validate positive finite or infinite coherence times."""
        _validate_lifetime(self.t1, name="t1")
        _validate_lifetime(self.t2_echo, name="t2_echo")


@dataclass(frozen=True)
class CrOnNoise:
    """Describe CR-active lifetimes in ns and transition rates in inverse ns."""

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
            _validate_rate(getattr(self, name), name=name)
        _validate_lifetime(self.target_t1rho, name="target_t1rho")


@dataclass(frozen=True)
class ZX90GateTiming:
    """Describe the semantic timing of an ideal echoed or un-echoed ZX90."""

    cr_lobe_duration: float
    echo: bool
    control_pi_duration: float = 0.0
    echo_margin_duration: float = 0.0

    def __post_init__(self) -> None:
        """Validate the semantic gate timing."""
        _validate_positive_finite(self.cr_lobe_duration, name="cr_lobe_duration")
        if not isinstance(self.echo, bool):
            raise TypeError("echo must be a boolean.")
        _validate_nonnegative_finite(
            self.control_pi_duration,
            name="control_pi_duration",
        )
        _validate_nonnegative_finite(
            self.echo_margin_duration,
            name="echo_margin_duration",
        )
        if not self.echo and (
            self.control_pi_duration > 0 or self.echo_margin_duration > 0
        ):
            raise ValueError(
                "An un-echoed ZX90 cannot contain an echo pi pulse or margin."
            )

    @property
    def duration(self) -> float:
        """Return the total semantic gate duration in ns."""
        if not self.echo:
            return self.cr_lobe_duration
        return 2 * (
            self.cr_lobe_duration
            + self.control_pi_duration
            + 2 * self.echo_margin_duration
        )


@dataclass(frozen=True)
class CrPulseFidelitySimulationResult:
    """Store the three fidelity limits and dissipative average leakage."""

    idle_coherence_limited_fidelity: float
    cr_on_coherence_limited_fidelity: float
    cr_on_dissipative_limited_fidelity: float
    average_leakage: float
    idle_average_survival: float
    cr_on_coherence_average_survival: float
    cr_on_dissipative_average_survival: float
    model_metadata: dict[str, object]


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


def _validate_rate(value: float, *, name: str) -> None:
    """Validate a nonnegative finite rate."""
    _validate_nonnegative_finite(value, name=name)


def extract_zx90_gate_timing(gate: PulseSchedule) -> ZX90GateTiming:
    """
    Extract semantic CR and echo timing from a Qubex ZX90 gate object.

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


def _collapse(rate: float, operator: NDArray[np.complex128]):
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


@dataclass(frozen=True)
class CrEchoDecayModel:
    """Cache the physical model used to predict both echo-decay protocols."""

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
        """Build one ZX90 map while varying only its fitted dephasing rates."""
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

    def predict(
        self,
        n_values: NDArray[np.int64],
        gamma_phi_control: float,
        gamma_phi_rho_target: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict the control-X and target-Z curves for specified rates."""
        gate_map = self.gate_map(gamma_phi_control, gamma_phi_rho_target)
        control_block = _compose(
            gate_map,
            self.control_x,
            gate_map,
            self.simultaneous_x,
            gate_map,
            self.control_x,
            gate_map,
        )
        target_block = _compose(gate_map, self.target_z, gate_map)

        plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
        ground = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        return (
            _expectation_curve(
                control_block,
                n_values,
                _state_vector(np.kron(plus, plus)),
                _CONTROL_X,
            ),
            _expectation_curve(
                target_block,
                2 * n_values,
                _state_vector(np.kron(ground, ground)),
                _TARGET_Z,
            ),
        )


def prepare_cr_echo_decay_model(
    timing: ZX90GateTiming,
    control_idle: IdleQubitNoise,
    target_idle: IdleQubitNoise,
    cr_noise: CrOnNoise,
    control_x180_duration: float,
    target_x180_duration: float,
) -> CrEchoDecayModel:
    """
    Prepare a reusable physical predictor for the C/D echo-decay curves.

    Every map and generator independent of the two variable CR-on dephasing
    rates is precomputed here. Measurement fitting is deliberately left to
    :func:`fit_cr_on_dephasing` in ``cr_pulse_coherence_fitting``.
    """
    if not timing.echo:
        raise ValueError("Echo-decay forward modeling requires an echoed ZX90 gate.")
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
        timing.control_pi_duration,
        np.pi,
        0.0,
        idle_operators,
    )
    echo_layer = _compose(margin, control_pi, margin)

    base_noise = replace(
        cr_noise,
        gamma_phi_control=0.0,
        gamma_phi_rho_target=0.0,
    )
    active_operators = _cr_collapse_operators(base_noise, include_leakage=True)
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
        positive_generator=_liouvillian(positive_hamiltonian, active_operators),
        negative_generator=_liouvillian(-positive_hamiltonian, active_operators),
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
        simultaneous_x=_x_layer_map(
            max(control_x180_duration, target_x180_duration),
            np.pi,
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
) -> NDArray[np.complex128]:
    """Build a semantic ZX90 map with CR-on/off noise switching."""
    idle_operators = (
        () if mode == "ideal" else _idle_collapse_operators(control_idle, target_idle)
    )
    if mode in ("ideal", "idle_coherence"):
        active_operators = idle_operators
    else:
        active_operators = _cr_collapse_operators(
            cr_noise,
            include_leakage=mode == "cr_on_dissipative",
        )

    lobe_angle = np.pi / 4 if timing.echo else np.pi / 2
    positive_cr = _segment_map(
        lobe_angle * _ZX / (2 * timing.cr_lobe_duration),
        timing.cr_lobe_duration,
        active_operators,
    )
    if not timing.echo:
        return positive_cr
    negative_cr = _segment_map(
        -lobe_angle * _ZX / (2 * timing.cr_lobe_duration),
        timing.cr_lobe_duration,
        active_operators,
    )
    margin = _segment_map(
        np.zeros((_FULL_DIMENSION,) * 2, dtype=np.complex128),
        timing.echo_margin_duration,
        idle_operators,
    )
    control_pi = _x_layer_map(
        timing.control_pi_duration,
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
) -> CrPulseFidelitySimulationResult:
    """
    Calculate three dissipative ZX90 fidelity limits and average leakage.

    The two qutrits are propagated with piecewise Lindblad superoperators.
    CR-active intervals use either idle noise or the supplied CR-on model;
    echo pulses and margins use idle noise. A noise-free simulation of the
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
    ideal_map = _zx90_map(timing, control_idle, target_idle, cr_noise, "ideal")
    idle_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "idle_coherence",
    )
    coherence_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "cr_on_coherence",
    )
    dissipative_map = _zx90_map(
        timing,
        control_idle,
        target_idle,
        cr_noise,
        "cr_on_dissipative",
    )
    idle_fidelity, idle_survival = _fidelity_and_survival(idle_map, ideal_map)
    coherence_fidelity, coherence_survival = _fidelity_and_survival(
        coherence_map,
        ideal_map,
    )
    dissipative_fidelity, dissipative_survival = _fidelity_and_survival(
        dissipative_map,
        ideal_map,
    )
    _, _, control_idle_clamped = _idle_rates(control_idle)
    _, _, target_idle_clamped = _idle_rates(target_idle)
    return CrPulseFidelitySimulationResult(
        idle_coherence_limited_fidelity=idle_fidelity,
        cr_on_coherence_limited_fidelity=coherence_fidelity,
        cr_on_dissipative_limited_fidelity=dissipative_fidelity,
        average_leakage=1 - dissipative_survival,
        idle_average_survival=idle_survival,
        cr_on_coherence_average_survival=coherence_survival,
        cr_on_dissipative_average_survival=dissipative_survival,
        model_metadata={
            "time_unit": "ns",
            "rate_unit": "1/ns",
            "t2_star_used": False,
            "coherent_errors_included": False,
            "coherent_leakage_included": False,
            "target_polarization_asymptote": 0.0,
            "target_seepage_return_model": "equal_incoherent_to_g_and_e",
            "reference_operation": "noise_off_same_semantic_gate",
            "gate_schedule_interpretation": "semantic_cr_echo_timing",
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
    values = np.empty(repetition_counts.shape, dtype=np.float64)
    state = initial_state
    previous_count = 0
    for index, count_value in enumerate(repetition_counts):
        count = int(count_value)
        if count < previous_count:
            raise ValueError("repetition counts must be nondecreasing.")
        for _ in range(count - previous_count):
            state = block @ state
        values[index] = _expectation(state, observable)
        previous_count = count
    return values


__all__ = [
    "CrEchoDecayModel",
    "CrOnNoise",
    "CrPulseFidelitySimulationResult",
    "IdleQubitNoise",
    "ZX90GateTiming",
    "extract_zx90_gate_timing",
    "prepare_cr_echo_decay_model",
    "simulate_cr_pulse_fidelity",
]
