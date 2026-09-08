"""Analyze measured CR-pulse coherence data and infer CR-on noise."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Literal, Protocol, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.linalg import expm
from scipy.optimize import least_squares

from qubex.pulse import PulseSchedule
from qubex.visualization import COLORS

from .cr_pulse_fidelity_simulation import (
    CrOnNoise,
    CrPulseFidelitySimulationResult,
    IdleQubitNoise,
    ZX90GateTiming,
    extract_zx90_gate_timing,
    prepare_cr_echo_decay_model,
    simulate_cr_pulse_fidelity,
)

TargetLeakageModel = Literal[
    "leakage_and_seepage",
    "leakage_only",
    "seepage_only",
    "none",
]
TargetT1RhoModel = Literal["exponential", "leakage_corrected"]

_RATE_BOUNDARY_FRACTION = 1e-5
_SINGLE_FORWARD_FIT_PARAMETER_COUNT = 3
_RATE_PARAMETER_COUNT = 4
_EXPONENTIAL_PARAMETER_COUNT = 3
_STATE_NAMES = ("g", "e", "f")
_CONTROL_GROUND = "control_ground"
_CONTROL_EXCITED = "control_excited"
_CONTROL_T2_ECHO = "control_t2_echo"
_TARGET_T2RHO_ECHO = "target_t2rho_echo"
_GEF_PROTOCOLS = (_CONTROL_GROUND, _CONTROL_EXCITED)
_PAULI_PROTOCOLS = (_CONTROL_T2_ECHO, _TARGET_T2RHO_ECHO)
_PRIMARY_PAULI = {_CONTROL_T2_ECHO: "X", _TARGET_T2RHO_ECHO: "Z"}
_PHENOMENOLOGICAL_ECHO_FITS = "phenomenological_echo_decay"
_REFERENCE_OPACITY = 0.38
_PROTOCOL_LABELS = {
    _CONTROL_GROUND: "Control initialized in |g>",
    _CONTROL_EXCITED: "Control initialized in |e>",
    _CONTROL_T2_ECHO: "Control T2 echo",
    _TARGET_T2RHO_ECHO: "Target T2rho echo",
}
_PAULI_MARKERS = {"X": "circle", "Y": "square", "Z": "triangle-up"}
_PAULI_REFERENCE_MARKERS = {
    "X": "diamond-open",
    "Y": "cross-open",
    "Z": "triangle-down-open",
}


class _CrControlEchoDecayModel(Protocol):
    """Describe the predictor required by the control-dephasing fit."""

    @property
    def cr_lobe_duration(self) -> float:
        """Return the CR-lobe duration used to scale fitted rates."""
        ...

    def predict_control_x(
        self: _CrControlEchoDecayModel,
        n_values: NDArray[np.int64],
        gamma_phi_control: float,
    ) -> NDArray[np.float64]:
        """Predict the control-X echo-decay curve."""
        ...


class _CrTargetEchoDecayModel(Protocol):
    """Describe the predictor required by the target-dephasing fit."""

    @property
    def cr_lobe_duration(self) -> float:
        """Return the CR-lobe duration used to scale fitted rates."""
        ...

    def predict_target_z(
        self: _CrTargetEchoDecayModel,
        n_values: NDArray[np.int64],
        gamma_phi_rho_target: float,
    ) -> NDArray[np.float64]:
        """Predict the target-Z echo-decay curve."""
        ...


def _positive_real(value: float | None, *, default: float, name: str) -> float:
    """Resolve and validate a positive finite real scalar."""
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return float(resolved)


@dataclass(frozen=True)
class TargetT1RhoFit:
    """Store a common-lifetime fit of two target-polarization curves."""

    success: bool
    message: str
    t1rho: float
    t1rho_error: float
    amplitude_ground: float
    amplitude_excited: float
    amplitude_ground_error: float
    amplitude_excited_error: float
    covariance: NDArray[np.float64]
    fitted_ground: NDArray[np.float64]
    fitted_excited: NDArray[np.float64]
    r_squared: float
    model: TargetT1RhoModel = "exponential"
    leakage_rate_used: float = 0.0


@dataclass(frozen=True)
class TargetLeakageFit:
    """Store a joint effective target leakage/seepage fit and fallback status."""

    success: bool
    message: str
    model: TargetLeakageModel | None
    leakage_rate: float
    seepage_rate: float
    leakage_rate_error: float
    seepage_rate_error: float
    covariance: NDArray[np.float64]
    initial_ground: float
    initial_excited: float
    fitted_ground: NDArray[np.float64]
    fitted_excited: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class CrOnDephasingFit:
    """Store one forward-fit of a CR-on dephasing rate."""

    success: bool
    message: str
    gamma_phi: float
    gamma_phi_error: float
    covariance: NDArray[np.float64]
    amplitude: float
    offset: float
    fitted_values: NDArray[np.float64]
    curve_n_values: NDArray[np.int64]
    curve_values: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class ExponentialDecayFit:
    """Store an offset exponential-decay fit and its diagnostics."""

    success: bool
    message: str
    amplitude: float
    offset: float
    tau: float
    amplitude_error: float
    offset_error: float
    tau_error: float
    covariance: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class ThreeLevelRateFit:
    """Store a joint ground/excited adjacent-transition three-level rate fit."""

    success: bool
    message: str
    leakage_model: Literal["full", "outward_only", "inward_only", "none"] | None
    gamma_ge_down: float
    gamma_ge_up: float
    gamma_ef_down: float
    gamma_ef_up: float
    gamma_ge_down_error: float
    gamma_ge_up_error: float
    gamma_ef_down_error: float
    gamma_ef_up_error: float
    t1_eff: float
    t1_eff_error: float
    covariance: NDArray[np.float64]
    initial_ground: NDArray[np.float64]
    initial_excited: NDArray[np.float64]
    fitted_ground: NDArray[np.float64]
    fitted_excited: NDArray[np.float64]
    r_squared: float


@dataclass(frozen=True)
class CrPulseCoherenceMeasurements:
    """
    Measured observables needed to repeat CR-coherence analysis offline.

    Times are in ns, rates inferred from them are in ``1/ns``, populations use
    the state order ``(g, e, f)``, and every observable contains separate
    ``actual`` and ``reference`` arrays.
    """

    control_qubit: str
    target_qubit: str
    protocols: tuple[str, ...]
    pauli_components: Mapping[str, tuple[str, ...]]
    n_values: tuple[int, ...]
    cr_pulse_counts: tuple[int, ...]
    times: Mapping[str, NDArray[np.float64]]
    sequence_durations: Mapping[str, NDArray[np.float64]]
    populations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    population_standard_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    target_polarizations: Mapping[str, Mapping[str, NDArray[np.float64]]]
    target_polarization_standard_errors: Mapping[
        str,
        Mapping[str, NDArray[np.float64]],
    ]
    pauli_expectations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    pauli_standard_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]


@dataclass(frozen=True)
class CrPulseFidelityAnalysis:
    """Independent CR-on dephasing fits and their optional simulation."""

    success: bool
    message: str
    cr_on_noise: CrOnNoise | None
    control_dephasing_fit: CrOnDephasingFit | None
    target_dephasing_fit: CrOnDephasingFit | None
    simulation: CrPulseFidelitySimulationResult | None


@dataclass(frozen=True)
class CrPulseCoherenceAnalysis:
    """Fits and derived quantities obtained from one measurement data set."""

    fits: Mapping[str, object]
    transition_rates: Mapping[str, object]
    decay_times: Mapping[str, object]
    fit_status: Mapping[str, object]
    base_cr_on_noise: CrOnNoise | None
    fidelity_analysis: CrPulseFidelityAnalysis | None


def _validate_measurement_array(
    values: ArrayLike,
    expected_shape: tuple[int, ...],
    *,
    name: str,
) -> None:
    """Validate the shape of one stored offline-analysis array."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real-valued array.") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")


def _validate_cr_pulse_coherence_measurements(
    measurements: CrPulseCoherenceMeasurements,
) -> None:
    """Validate the cross-field structure required for offline analysis."""
    if not isinstance(measurements, CrPulseCoherenceMeasurements):
        raise TypeError("measurements must be a CrPulseCoherenceMeasurements instance.")
    if (
        not isinstance(measurements.control_qubit, str)
        or not measurements.control_qubit
        or not isinstance(measurements.target_qubit, str)
        or not measurements.target_qubit
    ):
        raise ValueError("control_qubit and target_qubit must be nonempty strings.")
    if measurements.control_qubit == measurements.target_qubit:
        raise ValueError("control_qubit and target_qubit must be different.")

    protocols = measurements.protocols
    if not protocols or len(set(protocols)) != len(protocols):
        raise ValueError("protocols must be nonempty and contain no duplicates.")
    invalid_protocols = [
        protocol
        for protocol in protocols
        if protocol not in (*_GEF_PROTOCOLS, *_PAULI_PROTOCOLS)
    ]
    if invalid_protocols:
        raise ValueError(f"Unsupported protocol: {invalid_protocols[0]}.")
    if (_CONTROL_GROUND in protocols) != (_CONTROL_EXCITED in protocols):
        raise ValueError(
            "control_ground and control_excited must be supplied together."
        )

    n_values = measurements.n_values
    if len(n_values) < 3:
        raise ValueError("n_values must contain at least three values.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in n_values
    ):
        raise ValueError("n_values must contain only integers.")
    if n_values[0] != 0 or np.any(np.diff(np.asarray(n_values, dtype=np.int64)) <= 0):
        raise ValueError("n_values must increase strictly from zero.")
    expected_counts = tuple(4 * int(value) for value in n_values)
    if measurements.cr_pulse_counts != expected_counts:
        raise ValueError("cr_pulse_counts must equal 4 * n_values.")

    n_points = len(n_values)
    for protocol in protocols:
        try:
            stored_times = measurements.times[protocol]
        except KeyError as exc:
            raise ValueError(f"times is missing protocol {protocol}.") from exc
        if not isinstance(stored_times, np.ndarray):
            raise TypeError(f"times[{protocol!r}] must be a NumPy array.")
        times = np.asarray(stored_times, dtype=np.float64)
        if times.shape != (n_points,) or not np.all(np.isfinite(times)):
            raise ValueError(
                f"times[{protocol!r}] must contain {n_points} finite values."
            )
        if not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-12) or np.any(
            np.diff(times) <= 0
        ):
            raise ValueError(f"times[{protocol!r}] must increase strictly from zero.")
        for condition in (protocol, f"{protocol}_reference"):
            try:
                stored_durations = measurements.sequence_durations[condition]
            except KeyError as exc:
                raise ValueError(
                    f"sequence_durations is missing condition {condition}."
                ) from exc
            if not isinstance(stored_durations, np.ndarray):
                raise TypeError(
                    f"sequence_durations[{condition!r}] must be a NumPy array."
                )
            durations = np.asarray(stored_durations, dtype=np.float64)
            if (
                durations.shape != (n_points,)
                or not np.all(np.isfinite(durations))
                or np.any(durations < 0)
            ):
                raise ValueError(
                    f"sequence_durations[{condition!r}] must contain "
                    f"{n_points} nonnegative finite values."
                )
        if not np.allclose(
            measurements.sequence_durations[protocol],
            measurements.sequence_durations[f"{protocol}_reference"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{protocol} actual and reference sequence durations must match."
            )

    if all(protocol in protocols for protocol in _GEF_PROTOCOLS):
        if not np.allclose(
            measurements.times[_CONTROL_GROUND],
            measurements.times[_CONTROL_EXCITED],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "control_ground and control_excited must use the same time axis."
            )
        try:
            for protocol in _GEF_PROTOCOLS:
                for kind in ("actual", "reference"):
                    for qubit in (
                        measurements.control_qubit,
                        measurements.target_qubit,
                    ):
                        _validate_measurement_array(
                            measurements.populations[protocol][kind][qubit],
                            (n_points, len(_STATE_NAMES)),
                            name=f"populations[{protocol!r}][{kind!r}][{qubit!r}]",
                        )
                        _validate_measurement_array(
                            measurements.population_standard_errors[protocol][kind][
                                qubit
                            ],
                            (n_points, len(_STATE_NAMES)),
                            name=(
                                "population_standard_errors"
                                f"[{protocol!r}][{kind!r}][{qubit!r}]"
                            ),
                        )
                    _validate_measurement_array(
                        measurements.target_polarizations[protocol][kind],
                        (n_points,),
                        name=f"target_polarizations[{protocol!r}][{kind!r}]",
                    )
                    _validate_measurement_array(
                        measurements.target_polarization_standard_errors[protocol][
                            kind
                        ],
                        (n_points,),
                        name=(
                            "target_polarization_standard_errors"
                            f"[{protocol!r}][{kind!r}]"
                        ),
                    )
        except KeyError as exc:
            raise ValueError(
                f"GEF measurement data are missing key {exc.args[0]!r}."
            ) from exc

    try:
        for protocol in _PAULI_PROTOCOLS:
            if protocol not in protocols:
                continue
            components = measurements.pauli_components[protocol]
            primary = _PRIMARY_PAULI[protocol]
            if (
                not components
                or components[0] != primary
                or len(set(components)) != len(components)
                or any(component not in _PAULI_MARKERS for component in components)
            ):
                raise ValueError(
                    f"pauli_components[{protocol!r}] must contain unique X/Y/Z "
                    f"components with {primary} first."
                )
            for component in components:
                for kind in ("actual", "reference"):
                    _validate_measurement_array(
                        measurements.pauli_expectations[protocol][component][kind],
                        (n_points,),
                        name=(
                            f"pauli_expectations[{protocol!r}][{component!r}][{kind!r}]"
                        ),
                    )
                    _validate_measurement_array(
                        measurements.pauli_standard_errors[protocol][component][kind],
                        (n_points,),
                        name=(
                            f"pauli_standard_errors[{protocol!r}]"
                            f"[{component!r}][{kind!r}]"
                        ),
                    )
    except KeyError as exc:
        raise ValueError(
            f"Pauli measurement data are missing key {exc.args[0]!r}."
        ) from exc


def _validate_positive_finite(value: float, *, name: str) -> None:
    """Validate a positive finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number.")


def _validate_joint_series(
    times: ArrayLike,
    ground_values: ArrayLike,
    excited_values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate two finite one-dimensional curves on one time axis."""
    time_array = np.asarray(times, dtype=np.float64)
    ground = np.asarray(ground_values, dtype=np.float64)
    excited = np.asarray(excited_values, dtype=np.float64)
    if time_array.ndim != 1 or ground.ndim != 1 or excited.ndim != 1:
        raise ValueError("times and both value arrays must be one-dimensional.")
    if time_array.size < 3:
        raise ValueError("times must contain at least three points.")
    if ground.shape != time_array.shape or excited.shape != time_array.shape:
        raise ValueError("Both value arrays must have the same shape as times.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(ground)):
        raise ValueError("times and ground_values must contain only finite values.")
    if not np.all(np.isfinite(excited)):
        raise ValueError("excited_values must contain only finite values.")
    if not np.isclose(time_array[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("times must start at zero.")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("times must be strictly increasing.")
    return time_array, ground, excited


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    """Return a joint coefficient of determination."""
    residual_sum = float(np.sum((observed - fitted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else float("nan")
    return 1 - residual_sum / total_sum


def fit_target_t1rho(
    times: ArrayLike,
    polarization_ground: ArrayLike,
    polarization_excited: ArrayLike,
    standard_errors_ground: ArrayLike | None = None,
    standard_errors_excited: ArrayLike | None = None,
    *,
    f_population_ground: ArrayLike | None = None,
    f_population_excited: ArrayLike | None = None,
    leakage_rate: float = 0.0,
    relative_uncertainty_threshold: float = 1.0,
) -> TargetT1RhoFit:
    """
    Jointly fit two polarizations to separate amplitudes and one T1rho.

    By default, the fitted model is ``A * exp(-time / T1rho)``. If both
    F-population trajectories and the computational-to-F ``leakage_rate`` are
    supplied, the model removes the polarization dilution caused by incoherent
    seepage back into the computational subspace. The corrected model is

    ``A * q(0) / q(t) * exp(-(1 / T1rho + leakage_rate) * time)``,

    where ``q(t) = 1 - P_f(t)``. The supplied F-population trajectories should
    normally be smooth curves from the leakage/seepage fit rather than raw
    noisy observations. Their uncertainty and the uncertainty of
    ``leakage_rate`` are treated as fixed and are not propagated into the
    returned T1rho error.
    """
    time_array, ground, excited = _validate_joint_series(
        times,
        polarization_ground,
        polarization_excited,
    )
    errors_ground = _resolve_standard_errors(standard_errors_ground, ground.shape)
    errors_excited = _resolve_standard_errors(standard_errors_excited, excited.shape)
    _validate_positive_finite(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    if isinstance(leakage_rate, bool) or not isinstance(leakage_rate, Real):
        raise TypeError("leakage_rate must be a nonnegative finite real number.")
    if not np.isfinite(leakage_rate) or leakage_rate < 0:
        raise ValueError("leakage_rate must be a nonnegative finite real number.")
    has_ground_f = f_population_ground is not None
    has_excited_f = f_population_excited is not None
    if has_ground_f != has_excited_f:
        raise ValueError(
            "f_population_ground and f_population_excited must be supplied together."
        )
    if not has_ground_f and leakage_rate != 0:
        raise ValueError(
            "F-population trajectories are required when leakage_rate is nonzero."
        )

    model: TargetT1RhoModel = "exponential"
    correction_ground = np.ones_like(time_array)
    correction_excited = np.ones_like(time_array)
    if has_ground_f:
        _, pf_ground, pf_excited = _validate_joint_series(
            time_array,
            cast(ArrayLike, f_population_ground),
            cast(ArrayLike, f_population_excited),
        )
        if np.any((pf_ground < 0) | (pf_ground >= 1)) or np.any(
            (pf_excited < 0) | (pf_excited >= 1)
        ):
            raise ValueError("F-state populations must lie in [0, 1).")
        survival_ground = 1 - pf_ground
        survival_excited = 1 - pf_excited
        outward_survival = np.exp(-float(leakage_rate) * time_array)
        correction_ground = survival_ground[0] * outward_survival / survival_ground
        correction_excited = survival_excited[0] * outward_survival / survival_excited
        model = "leakage_corrected"
    weights_ground = (
        np.ones_like(ground) if errors_ground is None else 1 / errors_ground
    )
    weights_excited = (
        np.ones_like(excited) if errors_excited is None else 1 / errors_excited
    )
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    def curves(
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        amplitude_ground, amplitude_excited, scaled_t1rho = parameters
        decay = np.exp(-scaled_times / scaled_t1rho)
        return (
            amplitude_ground * correction_ground * decay,
            amplitude_excited * correction_excited * decay,
        )

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        fitted_ground, fitted_excited = curves(parameters)
        return np.concatenate(
            (
                (fitted_ground - ground) * weights_ground,
                (fitted_excited - excited) * weights_excited,
            )
        )

    optimization = least_squares(
        residual,
        x0=np.array([ground[0], excited[0], 0.5]),
        bounds=(
            np.array([-np.inf, -np.inf, np.finfo(float).eps]),
            np.array([np.inf, np.inf, np.inf]),
        ),
        max_nfev=20_000,
    )
    fitted_ground, fitted_excited = curves(optimization.x)
    scaled_covariance = _estimate_covariance(
        np.asarray(optimization.jac, dtype=np.float64),
        float(optimization.cost),
        2 * time_array.size,
        3,
        absolute_weights=errors_ground is not None or errors_excited is not None,
    )
    transform = np.diag([1.0, 1.0, time_scale])
    covariance = transform @ scaled_covariance @ transform
    parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    t1rho = float(optimization.x[2] * time_scale)
    identifiable = (
        np.linalg.matrix_rank(optimization.jac) == 3
        and np.isfinite(parameter_errors[2])
        and parameter_errors[2] / t1rho < relative_uncertainty_threshold
    )
    message = str(optimization.message)
    if not identifiable:
        message += " The common T1rho is not identifiable from these curves."
    observed = np.concatenate((ground, excited))
    fitted = np.concatenate((fitted_ground, fitted_excited))
    return TargetT1RhoFit(
        success=bool(optimization.success and identifiable),
        message=message,
        t1rho=t1rho,
        t1rho_error=float(parameter_errors[2]),
        amplitude_ground=float(optimization.x[0]),
        amplitude_excited=float(optimization.x[1]),
        amplitude_ground_error=float(parameter_errors[0]),
        amplitude_excited_error=float(parameter_errors[1]),
        covariance=covariance,
        fitted_ground=np.asarray(fitted_ground, dtype=np.float64),
        fitted_excited=np.asarray(fitted_excited, dtype=np.float64),
        r_squared=_r_squared(observed, fitted),
        model=model,
        leakage_rate_used=float(leakage_rate),
    )


def _target_leakage_curves(
    times: NDArray[np.float64],
    initial_ground: float,
    initial_excited: float,
    leakage_rate: float,
    seepage_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the shared effective computational-to-F rate model."""
    total_rate = leakage_rate + seepage_rate
    if total_rate == 0:
        return (
            np.full(times.shape, initial_ground),
            np.full(times.shape, initial_excited),
        )
    equilibrium = leakage_rate / total_rate
    decay = np.exp(-total_rate * times)
    return (
        equilibrium + (initial_ground - equilibrium) * decay,
        equilibrium + (initial_excited - equilibrium) * decay,
    )


def _parameter_is_resolved(
    value: float,
    standard_error: float,
    threshold: float,
    scale: float,
) -> bool:
    """Return whether a positive fitted rate has useful relative precision."""
    return bool(
        value * scale > 1e-8
        and np.isfinite(standard_error)
        and standard_error / value < threshold
    )


def fit_target_leakage(
    times: ArrayLike,
    pf_ground: ArrayLike,
    pf_excited: ArrayLike,
    standard_errors_ground: ArrayLike | None = None,
    standard_errors_excited: ArrayLike | None = None,
    *,
    relative_uncertainty_threshold: float = 1.0,
) -> TargetLeakageFit:
    """
    Jointly fit target leakage/seepage rates with fallback models.

    Both control preparations share
    ``dPf/dt = leakage_rate * (1-Pf) - seepage_rate * Pf`` while retaining
    independently measured initial F populations. If both rates cannot be
    identified, each one-rate reduced model is tested independently. If
    neither rate is resolved, the method returns a constant-curve model with
    both rates fixed to zero.
    """
    time_array, ground, excited = _validate_joint_series(
        times,
        pf_ground,
        pf_excited,
    )
    if np.any((ground < 0) | (ground > 1)) or np.any((excited < 0) | (excited > 1)):
        raise ValueError("Target F-state populations must lie in [0, 1].")
    _validate_positive_finite(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    errors_ground = _resolve_standard_errors(standard_errors_ground, ground.shape)
    errors_excited = _resolve_standard_errors(standard_errors_excited, excited.shape)
    weights_ground = (
        np.ones_like(ground) if errors_ground is None else 1 / errors_ground
    )
    weights_excited = (
        np.ones_like(excited) if errors_excited is None else 1 / errors_excited
    )
    initial_ground = float(ground[0])
    initial_excited = float(excited[0])
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    def optimize(active_indices: tuple[int, ...]):
        def residual(scaled_rates: NDArray[np.float64]) -> NDArray[np.float64]:
            rates = np.zeros(2, dtype=np.float64)
            rates[np.asarray(active_indices)] = scaled_rates
            fitted_ground, fitted_excited = _target_leakage_curves(
                scaled_times,
                initial_ground,
                initial_excited,
                float(rates[0]),
                float(rates[1]),
            )
            return np.concatenate(
                (
                    (fitted_ground - ground) * weights_ground,
                    (fitted_excited - excited) * weights_excited,
                )
            )

        initial_points = (
            ([0.01, 0.01], [0.1, 0.01], [0.01, 0.1], [0.5, 0.5])
            if len(active_indices) == 2
            else ([0.01], [0.1], [0.5])
        )
        candidates = [
            least_squares(
                residual,
                x0=np.asarray(point, dtype=np.float64),
                bounds=(0.0, np.inf),
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
                max_nfev=20_000,
            )
            for point in initial_points
        ]
        successful = [candidate for candidate in candidates if candidate.success]
        return min(successful or candidates, key=lambda candidate: candidate.cost)

    full_optimization = optimize((0, 1))
    full_covariance = (
        _estimate_covariance(
            np.asarray(full_optimization.jac, dtype=np.float64),
            float(full_optimization.cost),
            2 * time_array.size,
            2,
            absolute_weights=errors_ground is not None or errors_excited is not None,
        )
        / time_scale**2
    )
    full_errors = np.sqrt(np.clip(np.diag(full_covariance), 0.0, np.inf))
    full_rates = np.asarray(full_optimization.x, dtype=np.float64) / time_scale
    full_rate_sum = float(np.sum(full_rates))
    full_rank = np.linalg.matrix_rank(full_optimization.jac) == 2
    leakage_resolved = (
        full_optimization.success
        and full_rank
        and _parameter_is_resolved(
            float(full_rates[0]),
            float(full_errors[0]),
            relative_uncertainty_threshold,
            time_scale,
        )
        and full_rates[0] > _RATE_BOUNDARY_FRACTION * full_rate_sum
    )
    seepage_resolved = (
        full_optimization.success
        and full_rank
        and _parameter_is_resolved(
            float(full_rates[1]),
            float(full_errors[1]),
            relative_uncertainty_threshold,
            time_scale,
        )
        and full_rates[1] > _RATE_BOUNDARY_FRACTION * full_rate_sum
    )

    if leakage_resolved and seepage_resolved:
        model: TargetLeakageModel = "leakage_and_seepage"
        leakage_rate, seepage_rate = map(float, full_rates)
        leakage_error, seepage_error = map(float, full_errors)
        covariance = full_covariance
        fitted_ground, fitted_excited = _target_leakage_curves(
            time_array,
            initial_ground,
            initial_excited,
            leakage_rate,
            seepage_rate,
        )
        success = bool(full_optimization.success)
        message = str(full_optimization.message)
    else:
        reduced_candidates: list[
            tuple[float, TargetLeakageModel, int, float, float, str]
        ] = []
        reduced_models: tuple[tuple[int, TargetLeakageModel], ...]
        if leakage_resolved:
            reduced_models = ((0, "leakage_only"),)
        elif seepage_resolved:
            reduced_models = ((1, "seepage_only"),)
        else:
            reduced_models = ((0, "leakage_only"), (1, "seepage_only"))
        for rate_index, reduced_model in reduced_models:
            candidate = optimize((rate_index,))
            candidate_covariance = (
                _estimate_covariance(
                    np.asarray(candidate.jac, dtype=np.float64),
                    float(candidate.cost),
                    2 * time_array.size,
                    1,
                    absolute_weights=errors_ground is not None
                    or errors_excited is not None,
                )
                / time_scale**2
            )
            candidate_rate = float(candidate.x[0] / time_scale)
            candidate_error = float(np.sqrt(max(candidate_covariance[0, 0], 0.0)))
            if (
                candidate.success
                and np.linalg.matrix_rank(candidate.jac) == 1
                and _parameter_is_resolved(
                    candidate_rate,
                    candidate_error,
                    relative_uncertainty_threshold,
                    time_scale,
                )
            ):
                reduced_candidates.append(
                    (
                        float(candidate.cost),
                        reduced_model,
                        rate_index,
                        candidate_rate,
                        candidate_error,
                        str(candidate.message),
                    )
                )
        if reduced_candidates:
            (
                _,
                model,
                rate_index,
                selected_rate,
                selected_error,
                optimization_message,
            ) = min(reduced_candidates, key=lambda candidate: candidate[0])
            leakage_rate = selected_rate if rate_index == 0 else 0.0
            seepage_rate = selected_rate if rate_index == 1 else 0.0
            covariance = np.full((2, 2), np.nan)
            covariance[rate_index, rate_index] = selected_error**2
            leakage_error = selected_error if rate_index == 0 else float("nan")
            seepage_error = selected_error if rate_index == 1 else float("nan")
            fitted_ground, fitted_excited = _target_leakage_curves(
                time_array,
                initial_ground,
                initial_excited,
                leakage_rate,
                seepage_rate,
            )
            success = True
            fixed_rate = "seepage" if rate_index == 0 else "leakage"
            message = f"{optimization_message} Fallback: {fixed_rate} fixed to zero."
        else:
            model = "none"
            leakage_rate = seepage_rate = 0.0
            leakage_error = seepage_error = float("nan")
            covariance = np.full((2, 2), np.nan)
            fitted_ground, fitted_excited = _target_leakage_curves(
                time_array,
                initial_ground,
                initial_excited,
                0.0,
                0.0,
            )
            success = bool(full_optimization.success)
            message = (
                f"{full_optimization.message} Fallback: no resolved target "
                "leakage or seepage."
            )

    observed = np.concatenate((ground, excited))
    fitted = np.concatenate((fitted_ground, fitted_excited))
    return TargetLeakageFit(
        success=success,
        message=message,
        model=model,
        leakage_rate=leakage_rate,
        seepage_rate=seepage_rate,
        leakage_rate_error=leakage_error,
        seepage_rate_error=seepage_error,
        covariance=covariance,
        initial_ground=initial_ground,
        initial_excited=initial_excited,
        fitted_ground=np.asarray(fitted_ground, dtype=np.float64),
        fitted_excited=np.asarray(fitted_excited, dtype=np.float64),
        r_squared=_r_squared(observed, fitted),
    )


def _profile_affine(
    simulated: NDArray[np.float64],
    observed: NDArray[np.float64],
    errors: NDArray[np.float64] | None,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Profile an affine SPAM map and return its weighted residuals."""
    weights = np.ones_like(observed) if errors is None else 1 / errors
    design = np.column_stack((simulated, np.ones_like(simulated)))
    weighted_design = design * weights[:, None]
    amplitude, offset = np.linalg.lstsq(
        weighted_design,
        observed * weights,
        rcond=None,
    )[0]
    fitted = amplitude * simulated + offset
    return (
        float(amplitude),
        float(offset),
        fitted,
        (fitted - observed) * weights,
    )


def _fit_single_cr_on_dephasing(
    predictor: Callable[[NDArray[np.int64], float], NDArray[np.float64]],
    cr_lobe_duration: float,
    n_values: ArrayLike,
    values: ArrayLike,
    standard_errors: ArrayLike | None,
    *,
    value_name: str,
) -> CrOnDephasingFit:
    """Fit one CR-on dephasing rate while profiling an affine SPAM map."""
    n_array = np.asarray(n_values)
    observed = np.asarray(values, dtype=np.float64)
    if n_array.ndim != 1 or observed.shape != n_array.shape or n_array.size < 3:
        raise ValueError(
            f"n_values and {value_name} must be equal one-dimensional arrays "
            "with at least three values."
        )
    if not np.issubdtype(n_array.dtype, np.integer):
        raise ValueError("n_values must contain integers.")
    n_array = np.asarray(n_array, dtype=np.int64)
    if n_array[0] != 0 or np.any(np.diff(n_array) <= 0):
        raise ValueError("n_values must increase strictly from zero.")
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{value_name} must contain only finite values.")
    errors = _resolve_standard_errors(standard_errors, observed.shape)
    rate_scale = _positive_real(
        cr_lobe_duration,
        default=1.0,
        name="cr_lobe_duration",
    )

    def evaluate(
        scaled_rate: NDArray[np.float64],
    ) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        simulated = predictor(
            n_array,
            float(scaled_rate[0] / rate_scale),
        )
        return _profile_affine(
            np.asarray(simulated, dtype=np.float64),
            observed,
            errors,
        )

    def residual(scaled_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return evaluate(scaled_rate)[3]

    optimization = least_squares(
        residual,
        x0=np.array([1e-3], dtype=np.float64),
        bounds=(0.0, np.inf),
        x_scale=1.0,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        max_nfev=80,
    )
    profile = evaluate(optimization.x)
    covariance = (
        _estimate_covariance(
            np.asarray(optimization.jac, dtype=np.float64),
            float(optimization.cost),
            n_array.size,
            _SINGLE_FORWARD_FIT_PARAMETER_COUNT,
            absolute_weights=errors is not None,
        )
        / rate_scale**2
    )
    gamma_phi_error = float(np.sqrt(np.clip(covariance[0, 0], 0.0, np.inf)))
    dense_n = np.arange(int(n_array[-1]) + 1, dtype=np.int64)
    dense_simulated = predictor(
        dense_n,
        float(optimization.x[0] / rate_scale),
    )
    curve_values = profile[0] * dense_simulated + profile[1]
    identifiable = np.linalg.matrix_rank(optimization.jac) == 1
    message = str(optimization.message)
    if not identifiable:
        message += " The CR-on dephasing rate is not identifiable."
    return CrOnDephasingFit(
        success=bool(optimization.success and identifiable),
        message=message,
        gamma_phi=float(optimization.x[0] / rate_scale),
        gamma_phi_error=gamma_phi_error,
        covariance=covariance,
        amplitude=profile[0],
        offset=profile[1],
        fitted_values=np.asarray(profile[2], dtype=np.float64),
        curve_n_values=dense_n,
        curve_values=np.asarray(curve_values, dtype=np.float64),
        r_squared=_r_squared(observed, profile[2]),
    )


def fit_cr_on_control_dephasing(
    model: _CrControlEchoDecayModel,
    n_values: ArrayLike,
    control_x: ArrayLike,
    standard_errors: ArrayLike | None = None,
) -> CrOnDephasingFit:
    """
    Infer the control CR-on dephasing rate from a control-X echo curve.

    The target CR-on dephasing rate is fixed to zero by the model prediction.
    An affine amplitude and offset are profiled to absorb SPAM.

    Parameters
    ----------
    model
        Physical predictor for the control echo protocol.
    n_values
        Strictly increasing nonnegative repetition indices beginning at zero.
    control_x
        Measured control-X expectation values.
    standard_errors
        Optional one-standard-error uncertainties used as absolute weights.

    Returns
    -------
    CrOnDephasingFit
        Fitted `gamma_phi_control` in inverse time units of the model.
    """
    return _fit_single_cr_on_dephasing(
        model.predict_control_x,
        model.cr_lobe_duration,
        n_values,
        control_x,
        standard_errors,
        value_name="control_x",
    )


def fit_cr_on_target_rotating_frame_dephasing(
    model: _CrTargetEchoDecayModel,
    n_values: ArrayLike,
    target_z: ArrayLike,
    standard_errors: ArrayLike | None = None,
) -> CrOnDephasingFit:
    """
    Infer target CR-on rotating-frame dephasing from a target-Z echo curve.

    The control CR-on dephasing rate is fixed to zero by the model prediction.
    An affine amplitude and offset are profiled to absorb SPAM.

    Parameters
    ----------
    model
        Physical predictor for the target rotating-frame echo protocol.
    n_values
        Strictly increasing nonnegative repetition indices beginning at zero.
    target_z
        Measured target-Z expectation values.
    standard_errors
        Optional one-standard-error uncertainties used as absolute weights.

    Returns
    -------
    CrOnDephasingFit
        Fitted `gamma_phi_rho_target` in inverse time units of the model.
    """
    return _fit_single_cr_on_dephasing(
        model.predict_target_z,
        model.cr_lobe_duration,
        n_values,
        target_z,
        standard_errors,
        value_name="target_z",
    )


def _validate_time_series(
    times: ArrayLike,
    values: ArrayLike,
    *,
    name: str,
    minimum_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a finite series sampled on an increasing zero-based time axis."""
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if time_array.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if value_array.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional.")
    if value_array.shape[0] != time_array.size:
        raise ValueError(f"{name} must have the same leading length as times.")
    if time_array.size < minimum_points:
        raise ValueError(f"times must contain at least {minimum_points} points.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError(f"times and {name} must contain only finite values.")
    if not np.isclose(time_array[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("times must start at zero.")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("times must be strictly increasing.")
    return time_array, value_array


def _resolve_standard_errors(
    standard_errors: ArrayLike | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Validate uncertainties and floor zero values for stable weighting."""
    if standard_errors is None:
        return None
    errors = np.asarray(standard_errors, dtype=np.float64)
    if errors.shape != shape:
        raise ValueError(f"standard_errors must have shape {shape}.")
    if np.any(np.isfinite(errors) & (errors < 0)):
        raise ValueError("standard_errors must be nonnegative where finite.")
    positive = errors[np.isfinite(errors) & (errors > 0)]
    if positive.size == 0:
        return None
    typical = float(np.median(positive))
    floor = max(typical * 1e-3, np.finfo(float).eps)
    return np.where(
        np.isfinite(errors) & (errors > 0),
        np.maximum(errors, floor),
        typical,
    )


def _estimate_covariance(
    jacobian: NDArray[np.float64],
    cost: float,
    residual_count: int,
    parameter_count: int,
    *,
    absolute_weights: bool,
) -> NDArray[np.float64]:
    """Estimate covariance using absolute weights or fitted residual variance."""
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    if absolute_weights:
        return covariance
    degrees_of_freedom = residual_count - parameter_count
    if degrees_of_freedom > 0:
        covariance *= 2 * cost / degrees_of_freedom
    else:
        covariance.fill(np.nan)
    return covariance


def fit_exponential_decay(
    times: ArrayLike,
    values: ArrayLike,
    standard_errors: ArrayLike | None = None,
) -> ExponentialDecayFit:
    """
    Fit `offset + amplitude * exp(-time / tau)` to a decay series.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    values
        Finite measured values.
    standard_errors
        Optional nonnegative one-standard-error uncertainties used as absolute
        fit weights. Missing or zero entries use the median positive error when
        available; otherwise the fit is unweighted.

    Returns
    -------
    ExponentialDecayFit
        Fitted parameters. `tau` uses the same unit as `times`.
    """
    time_array, value_array = _validate_time_series(
        times,
        values,
        name="values",
        minimum_points=_EXPONENTIAL_PARAMETER_COUNT,
    )
    if value_array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    errors = _resolve_standard_errors(standard_errors, value_array.shape)
    weights = np.ones_like(value_array) if errors is None else 1 / errors
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    offset_guess = float(value_array[-1])
    amplitude_guess = float(value_array[0] - offset_guess)
    if np.isclose(amplitude_guess, 0.0):
        amplitude_guess = float(np.ptp(value_array))

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        amplitude, offset, scaled_tau = parameters
        prediction = offset + amplitude * np.exp(-scaled_times / scaled_tau)
        return (prediction - value_array) * weights

    try:
        optimization = least_squares(
            residual,
            x0=np.array([amplitude_guess, offset_guess, 0.5]),
            bounds=(
                np.array([-np.inf, -np.inf, np.finfo(float).eps]),
                np.array([np.inf, np.inf, np.inf]),
            ),
            x_scale=1.0,
            max_nfev=20_000,
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        return _failed_exponential_fit(str(exc), value_array.size)

    amplitude, offset, scaled_tau = optimization.x
    tau = float(scaled_tau * time_scale)
    fitted_values = offset + amplitude * np.exp(-time_array / tau)
    scaled_covariance = _estimate_covariance(
        np.asarray(optimization.jac, dtype=np.float64),
        float(optimization.cost),
        value_array.size,
        _EXPONENTIAL_PARAMETER_COUNT,
        absolute_weights=errors is not None,
    )
    transform = np.diag([1.0, 1.0, time_scale])
    covariance = transform @ scaled_covariance @ transform
    errors_by_parameter = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return ExponentialDecayFit(
        success=bool(optimization.success),
        message=str(optimization.message),
        amplitude=float(amplitude),
        offset=float(offset),
        tau=tau,
        amplitude_error=float(errors_by_parameter[0]),
        offset_error=float(errors_by_parameter[1]),
        tau_error=float(errors_by_parameter[2]),
        covariance=covariance,
        fitted_values=np.asarray(fitted_values, dtype=np.float64),
        r_squared=_r_squared(value_array, fitted_values),
    )


def _failed_exponential_fit(message: str, n_values: int) -> ExponentialDecayFit:
    """Return a structured failed exponential fit."""
    return ExponentialDecayFit(
        success=False,
        message=message,
        amplitude=float("nan"),
        offset=float("nan"),
        tau=float("nan"),
        amplitude_error=float("nan"),
        offset_error=float("nan"),
        tau_error=float("nan"),
        covariance=np.full((_EXPONENTIAL_PARAMETER_COUNT,) * 2, np.nan),
        fitted_values=np.full(n_values, np.nan),
        r_squared=float("nan"),
    )


def _three_level_rate_matrix(
    rates: ArrayLike,
) -> NDArray[np.float64]:
    """Return the GEF rate matrix with no direct G-to-F transition."""
    gamma_ge_down, gamma_ge_up, gamma_ef_down, gamma_ef_up = np.asarray(
        rates,
        dtype=np.float64,
    )
    return np.array(
        [
            [-gamma_ge_up, gamma_ge_down, 0.0],
            [
                gamma_ge_up,
                -(gamma_ge_down + gamma_ef_up),
                gamma_ef_down,
            ],
            [0.0, gamma_ef_up, -gamma_ef_down],
        ],
        dtype=np.float64,
    )


def three_level_population_trajectory(
    times: NDArray[np.float64],
    initial_population: NDArray[np.float64],
    rates: ArrayLike,
) -> NDArray[np.float64]:
    """Propagate a GEF population under the adjacent-transition rate model."""
    matrix = _three_level_rate_matrix(rates)
    return np.stack([expm(matrix * float(time)) @ initial_population for time in times])


def _validate_population_series(
    populations: NDArray[np.float64],
    *,
    name: str,
) -> None:
    """Validate physical GEF population rows."""
    if populations.ndim != 2 or populations.shape[1] != len(_STATE_NAMES):
        raise ValueError(f"{name} must have shape (n_times, 3).")
    tolerance = 1e-7
    if np.any(populations < -tolerance) or np.any(populations > 1 + tolerance):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    if not np.allclose(np.sum(populations, axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Rows of {name} must sum to one.")


def _failed_rate_fit(
    message: str,
    initial_ground: NDArray[np.float64],
    initial_excited: NDArray[np.float64],
    n_times: int,
) -> ThreeLevelRateFit:
    """Return a structured failed rate-model fit."""
    return ThreeLevelRateFit(
        success=False,
        message=message,
        leakage_model=None,
        gamma_ge_down=float("nan"),
        gamma_ge_up=float("nan"),
        gamma_ef_down=float("nan"),
        gamma_ef_up=float("nan"),
        gamma_ge_down_error=float("nan"),
        gamma_ge_up_error=float("nan"),
        gamma_ef_down_error=float("nan"),
        gamma_ef_up_error=float("nan"),
        t1_eff=float("nan"),
        t1_eff_error=float("nan"),
        covariance=np.full((_RATE_PARAMETER_COUNT,) * 2, np.nan),
        initial_ground=initial_ground,
        initial_excited=initial_excited,
        fitted_ground=np.full((n_times, len(_STATE_NAMES)), np.nan),
        fitted_excited=np.full((n_times, len(_STATE_NAMES)), np.nan),
        r_squared=float("nan"),
    )


def fit_three_level_rate_model(
    times: ArrayLike,
    populations_ground: ArrayLike,
    populations_excited: ArrayLike,
    standard_errors_ground: ArrayLike | None = None,
    standard_errors_excited: ArrayLike | None = None,
    *,
    relative_uncertainty_threshold: float = 1.0,
) -> ThreeLevelRateFit:
    """
    Jointly fit ground/excited initial states to a four-rate adjacent GEF model.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    populations_ground
        Populations initialized near g, ordered as g, e, f, with shape
        `(n_times, 3)`.
    populations_excited
        Populations initialized near e, ordered as g, e, f, with shape
        `(n_times, 3)`.
    standard_errors_ground
        Optional nonnegative one-standard-error uncertainties for
        `populations_ground`, used as absolute fit weights.
    standard_errors_excited
        Optional nonnegative one-standard-error uncertainties for
        `populations_excited`, used as absolute fit weights.
    relative_uncertainty_threshold
        Maximum relative uncertainty used to retain an independently fitted
        leakage or seepage rate. Unresolved rates are fixed to zero and the
        reduced model is refitted.

    Returns
    -------
    ThreeLevelRateFit
        Directed rates and `T1_eff`. Rates are inverse `times` units. The
        `leakage_model` records whether the full, outward-only, inward-only,
        or no-leakage model was identifiable and used for the returned curves.
    """
    time_array, population_ground = _validate_time_series(
        times,
        populations_ground,
        name="populations_ground",
        minimum_points=3,
    )
    _, population_excited = _validate_time_series(
        times,
        populations_excited,
        name="populations_excited",
        minimum_points=3,
    )
    _validate_population_series(population_ground, name="populations_ground")
    _validate_population_series(population_excited, name="populations_excited")
    threshold = _positive_real(
        relative_uncertainty_threshold,
        default=1.0,
        name="relative_uncertainty_threshold",
    )

    errors_ground = _resolve_standard_errors(
        standard_errors_ground, population_ground.shape
    )
    errors_excited = _resolve_standard_errors(
        standard_errors_excited, population_excited.shape
    )
    weights_ground = (
        np.ones_like(population_ground) if errors_ground is None else 1 / errors_ground
    )
    weights_excited = (
        np.ones_like(population_excited)
        if errors_excited is None
        else 1 / errors_excited
    )
    initial_ground = np.clip(population_ground[0], 0.0, 1.0)
    initial_excited = np.clip(population_excited[0], 0.0, 1.0)
    initial_ground /= np.sum(initial_ground)
    initial_excited /= np.sum(initial_excited)
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    def rates_from_active(
        active_rates: NDArray[np.float64],
        active_indices: tuple[int, ...],
    ) -> NDArray[np.float64]:
        rates = np.zeros(_RATE_PARAMETER_COUNT, dtype=np.float64)
        rates[np.asarray(active_indices)] = active_rates
        return rates

    def trajectories(
        active_rates: NDArray[np.float64],
        active_indices: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        scaled_rates = rates_from_active(active_rates, active_indices)
        return (
            three_level_population_trajectory(
                scaled_times, initial_ground, scaled_rates
            ),
            three_level_population_trajectory(
                scaled_times, initial_excited, scaled_rates
            ),
        )

    def optimize(active_indices: tuple[int, ...]):
        def residual(active_rates: NDArray[np.float64]) -> NDArray[np.float64]:
            fitted_ground, fitted_excited = trajectories(
                active_rates,
                active_indices,
            )
            return np.concatenate(
                (
                    ((fitted_ground - population_ground) * weights_ground).ravel(),
                    ((fitted_excited - population_excited) * weights_excited).ravel(),
                )
            )

        def attempt(initial_scale: float):
            try:
                return least_squares(
                    residual,
                    x0=np.full(len(active_indices), initial_scale),
                    bounds=(0.0, np.inf),
                    x_scale=1.0,
                    ftol=1e-13,
                    xtol=1e-13,
                    gtol=1e-13,
                    max_nfev=30_000,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                return None

        attempts = [
            result
            for initial_scale in (0.05, 0.2, 0.7, 2.0)
            if (result := attempt(initial_scale)) is not None
        ]
        if not attempts:
            return None
        successful_attempts = [candidate for candidate in attempts if candidate.success]
        return min(
            successful_attempts or attempts, key=lambda candidate: candidate.cost
        )

    full_indices = (0, 1, 2, 3)
    full_optimization = optimize(full_indices)
    if full_optimization is None:
        return _failed_rate_fit(
            "All least-squares attempts failed.",
            initial_ground,
            initial_excited,
            time_array.size,
        )

    residual_count = population_ground.size + population_excited.size
    full_scaled_covariance = _estimate_covariance(
        np.asarray(full_optimization.jac, dtype=np.float64),
        float(full_optimization.cost),
        residual_count,
        _RATE_PARAMETER_COUNT,
        absolute_weights=errors_ground is not None or errors_excited is not None,
    )
    full_covariance = full_scaled_covariance / time_scale**2
    full_rates = np.asarray(full_optimization.x, dtype=np.float64) / time_scale
    full_errors = np.sqrt(np.clip(np.diag(full_covariance), 0.0, np.inf))
    ef_rate_sum = float(full_rates[2] + full_rates[3])
    full_rank = np.linalg.matrix_rank(full_optimization.jac) == _RATE_PARAMETER_COUNT
    outward_resolved = (
        full_optimization.success
        and full_rank
        and (
            full_optimization.x[3] > 1e-8
            and np.isfinite(full_errors[3])
            and full_errors[3] / full_rates[3] < threshold
            and full_rates[3] > _RATE_BOUNDARY_FRACTION * ef_rate_sum
        )
    )
    seepage_resolved = (
        full_optimization.success
        and full_rank
        and (
            full_optimization.x[2] > 1e-8
            and np.isfinite(full_errors[2])
            and full_errors[2] / full_rates[2] < threshold
            and full_rates[2] > _RATE_BOUNDARY_FRACTION * ef_rate_sum
        )
    )
    if outward_resolved and seepage_resolved:
        leakage_model = "full"
        active_indices = full_indices
        fallback_message = ""
        optimization = full_optimization
    else:
        reduced_candidates = []
        if outward_resolved:
            reduced_models = ((3, "outward_only"),)
        elif seepage_resolved:
            reduced_models = ((2, "inward_only"),)
        else:
            reduced_models = ((3, "outward_only"), (2, "inward_only"))
        for rate_index, reduced_model in reduced_models:
            candidate_indices = (0, 1, rate_index)
            candidate = optimize(candidate_indices)
            if candidate is None:
                continue
            candidate_covariance = (
                _estimate_covariance(
                    np.asarray(candidate.jac, dtype=np.float64),
                    float(candidate.cost),
                    residual_count,
                    len(candidate_indices),
                    absolute_weights=errors_ground is not None
                    or errors_excited is not None,
                )
                / time_scale**2
            )
            candidate_rate = float(candidate.x[2] / time_scale)
            candidate_error = float(np.sqrt(max(candidate_covariance[2, 2], 0.0)))
            if (
                candidate.success
                and np.linalg.matrix_rank(candidate.jac) == len(candidate_indices)
                and _parameter_is_resolved(
                    candidate_rate,
                    candidate_error,
                    threshold,
                    time_scale,
                )
            ):
                reduced_candidates.append(
                    (
                        float(candidate.cost),
                        reduced_model,
                        candidate_indices,
                        candidate,
                    )
                )
        if reduced_candidates:
            _, leakage_model, active_indices, optimization = min(
                reduced_candidates,
                key=lambda candidate: candidate[0],
            )
            fixed_transition = (
                "F-to-E seepage"
                if leakage_model == "outward_only"
                else "E-to-F leakage"
            )
            fallback_message = f"Unresolved control {fixed_transition} fixed to zero."
        else:
            leakage_model = "none"
            active_indices = (0, 1)
            fallback_message = (
                "Control E-to-F leakage and F-to-E seepage fixed to zero."
            )
            optimization = optimize(active_indices)
    if optimization is None:
        return _failed_rate_fit(
            "The selected reduced rate-model fit failed.",
            initial_ground,
            initial_excited,
            time_array.size,
        )
    scaled_rates = rates_from_active(
        np.asarray(optimization.x, dtype=np.float64),
        active_indices,
    )
    rates = scaled_rates / time_scale
    fitted_ground, fitted_excited = (
        three_level_population_trajectory(time_array, initial_ground, rates),
        three_level_population_trajectory(time_array, initial_excited, rates),
    )
    active_covariance = (
        _estimate_covariance(
            np.asarray(optimization.jac, dtype=np.float64),
            float(optimization.cost),
            residual_count,
            len(active_indices),
            absolute_weights=errors_ground is not None or errors_excited is not None,
        )
        / time_scale**2
    )
    covariance = np.full((_RATE_PARAMETER_COUNT,) * 2, np.nan)
    covariance[np.ix_(active_indices, active_indices)] = active_covariance
    rate_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    ge_rate_sum = float(rates[0] + rates[1])
    t1_eff = 1 / ge_rate_sum if ge_rate_sum > 0 else float("inf")
    ge_rate_sum_variance = float(
        covariance[0, 0] + covariance[1, 1] + 2 * covariance[0, 1]
    )
    t1_eff_error = (
        np.sqrt(max(ge_rate_sum_variance, 0.0)) / ge_rate_sum**2
        if ge_rate_sum > 0
        else float("nan")
    )
    observed = np.concatenate((population_ground.ravel(), population_excited.ravel()))
    fitted = np.concatenate((fitted_ground.ravel(), fitted_excited.ravel()))
    selected_model_identifiable = np.linalg.matrix_rank(optimization.jac) == len(
        active_indices
    )
    fit_message = " ".join(
        part for part in (str(optimization.message), fallback_message) if part
    )
    if not selected_model_identifiable:
        fit_message += " The selected rate model is rank deficient."
    return ThreeLevelRateFit(
        success=bool(optimization.success and selected_model_identifiable),
        message=fit_message,
        leakage_model=leakage_model,
        gamma_ge_down=float(rates[0]),
        gamma_ge_up=float(rates[1]),
        gamma_ef_down=float(rates[2]),
        gamma_ef_up=float(rates[3]),
        gamma_ge_down_error=float(rate_errors[0]),
        gamma_ge_up_error=float(rate_errors[1]),
        gamma_ef_down_error=float(rate_errors[2]),
        gamma_ef_up_error=float(rate_errors[3]),
        t1_eff=float(t1_eff),
        t1_eff_error=float(t1_eff_error),
        covariance=covariance,
        initial_ground=initial_ground,
        initial_excited=initial_excited,
        fitted_ground=fitted_ground,
        fitted_excited=fitted_excited,
        r_squared=_r_squared(observed, fitted),
    )


def _finite_errors_or_none(
    errors: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Return uncertainty data only when at least one value is usable."""
    return errors if np.any(np.isfinite(errors) & (errors > 0)) else None


def _safe_fit_target_t1rho(
    times: NDArray[np.float64],
    ground: NDArray[np.float64],
    excited: NDArray[np.float64],
    ground_errors: NDArray[np.float64] | None,
    excited_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
    *,
    f_population_ground: NDArray[np.float64] | None = None,
    f_population_excited: NDArray[np.float64] | None = None,
    leakage_rate: float = 0.0,
) -> TargetT1RhoFit:
    """Fit target T1rho without discarding data after a numerical failure."""
    try:
        return fit_target_t1rho(
            times,
            ground,
            excited,
            ground_errors,
            excited_errors,
            f_population_ground=f_population_ground,
            f_population_excited=f_population_excited,
            leakage_rate=leakage_rate,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return TargetT1RhoFit(
            success=False,
            message=str(exc),
            t1rho=float("nan"),
            t1rho_error=float("nan"),
            amplitude_ground=float("nan"),
            amplitude_excited=float("nan"),
            amplitude_ground_error=float("nan"),
            amplitude_excited_error=float("nan"),
            covariance=np.full((3, 3), np.nan),
            fitted_ground=np.full(times.shape, np.nan),
            fitted_excited=np.full(times.shape, np.nan),
            r_squared=float("nan"),
            model=(
                "leakage_corrected"
                if f_population_ground is not None and f_population_excited is not None
                else "exponential"
            ),
            leakage_rate_used=leakage_rate,
        )


def _safe_fit_target_leakage(
    times: NDArray[np.float64],
    ground: NDArray[np.float64],
    excited: NDArray[np.float64],
    ground_errors: NDArray[np.float64] | None,
    excited_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> TargetLeakageFit:
    """Fit target leakage without discarding data after numerical failure."""
    try:
        return fit_target_leakage(
            times,
            ground,
            excited,
            ground_errors,
            excited_errors,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return TargetLeakageFit(
            success=False,
            message=str(exc),
            model=None,
            leakage_rate=float("nan"),
            seepage_rate=float("nan"),
            leakage_rate_error=float("nan"),
            seepage_rate_error=float("nan"),
            covariance=np.full((2, 2), np.nan),
            initial_ground=float(ground[0]),
            initial_excited=float(excited[0]),
            fitted_ground=np.full(times.shape, np.nan),
            fitted_excited=np.full(times.shape, np.nan),
            r_squared=float("nan"),
        )


def _safe_fit_three_level_rate_model(
    times: NDArray[np.float64],
    populations_ground: NDArray[np.float64],
    populations_excited: NDArray[np.float64],
    standard_errors_ground: NDArray[np.float64] | None,
    standard_errors_excited: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> ThreeLevelRateFit:
    """Fit the control rate model and represent numerical failure explicitly."""
    try:
        return fit_three_level_rate_model(
            times,
            populations_ground,
            populations_excited,
            standard_errors_ground,
            standard_errors_excited,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_rate_fit(
            str(exc),
            np.asarray(populations_ground[0], dtype=np.float64),
            np.asarray(populations_excited[0], dtype=np.float64),
            times.size,
        )


def _safe_fit_exponential_decay(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
) -> ExponentialDecayFit:
    """Fit an exponential and represent numerical failure explicitly."""
    try:
        return fit_exponential_decay(times, values, standard_errors)
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_exponential_fit(str(exc), values.size)


def _fit_measured_observables(
    measurements: CrPulseCoherenceMeasurements,
    relative_uncertainty_threshold: float,
) -> dict[str, object]:
    """Fit all empirical models supported by the available protocols."""
    fits: dict[str, object] = {}
    protocols = measurements.protocols
    if all(protocol in protocols for protocol in _GEF_PROTOCOLS):
        control = measurements.control_qubit
        target = measurements.target_qubit
        times = measurements.times[_CONTROL_GROUND]
        populations = measurements.populations
        errors = measurements.population_standard_errors
        fits["control_rate_model"] = {
            kind: _safe_fit_three_level_rate_model(
                times,
                populations[_CONTROL_GROUND][kind][control],
                populations[_CONTROL_EXCITED][kind][control],
                _finite_errors_or_none(errors[_CONTROL_GROUND][kind][control]),
                _finite_errors_or_none(errors[_CONTROL_EXCITED][kind][control]),
                relative_uncertainty_threshold,
            )
            for kind in ("actual", "reference")
        }
        target_leakage_fits = {
            kind: _safe_fit_target_leakage(
                times,
                populations[_CONTROL_GROUND][kind][target][:, 2],
                populations[_CONTROL_EXCITED][kind][target][:, 2],
                _finite_errors_or_none(errors[_CONTROL_GROUND][kind][target][:, 2]),
                _finite_errors_or_none(errors[_CONTROL_EXCITED][kind][target][:, 2]),
                relative_uncertainty_threshold,
            )
            for kind in ("actual", "reference")
        }
        fits["target_leakage"] = target_leakage_fits
        fits["target_t1rho"] = {
            kind: _safe_fit_target_t1rho(
                times,
                measurements.target_polarizations[_CONTROL_GROUND][kind],
                measurements.target_polarizations[_CONTROL_EXCITED][kind],
                _finite_errors_or_none(
                    measurements.target_polarization_standard_errors[_CONTROL_GROUND][
                        kind
                    ]
                ),
                _finite_errors_or_none(
                    measurements.target_polarization_standard_errors[_CONTROL_EXCITED][
                        kind
                    ]
                ),
                relative_uncertainty_threshold,
                f_population_ground=(
                    target_leakage_fits[kind].fitted_ground
                    if target_leakage_fits[kind].success
                    else None
                ),
                f_population_excited=(
                    target_leakage_fits[kind].fitted_excited
                    if target_leakage_fits[kind].success
                    else None
                ),
                leakage_rate=(
                    target_leakage_fits[kind].leakage_rate
                    if target_leakage_fits[kind].success
                    else 0.0
                ),
            )
            for kind in ("actual", "reference")
        }

    echo_fits = {
        protocol: {
            kind: _safe_fit_exponential_decay(
                measurements.times[protocol],
                measurements.pauli_expectations[protocol][_PRIMARY_PAULI[protocol]][
                    kind
                ],
                _finite_errors_or_none(
                    measurements.pauli_standard_errors[protocol][
                        _PRIMARY_PAULI[protocol]
                    ][kind]
                ),
            )
            for kind in ("actual", "reference")
        }
        for protocol in _PAULI_PROTOCOLS
        if protocol in protocols
    }
    if echo_fits:
        fits[_PHENOMENOLOGICAL_ECHO_FITS] = echo_fits
    return fits


def _value_with_error(value: float, standard_error: float) -> dict[str, float]:
    """Return one estimate and its one-standard-error uncertainty."""
    return {"value": value, "standard_error": standard_error}


def _summarize_fit_parameters(
    fits: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build compact rate and decay-time summaries from fit objects."""
    transition_rates: dict[str, object] = {"unit": "1/ns"}
    decay_times: dict[str, object] = {"unit": "ns"}
    if "control_rate_model" in fits:
        rate_fits = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])
        t1rho_fits = cast(Mapping[str, TargetT1RhoFit], fits["target_t1rho"])
        leakage_fits = cast(Mapping[str, TargetLeakageFit], fits["target_leakage"])
        transition_rates.update(
            {
                kind: {
                    "gamma_ge_down": _value_with_error(
                        fit.gamma_ge_down, fit.gamma_ge_down_error
                    ),
                    "gamma_ge_up": _value_with_error(
                        fit.gamma_ge_up, fit.gamma_ge_up_error
                    ),
                    "gamma_ef_down": _value_with_error(
                        fit.gamma_ef_down, fit.gamma_ef_down_error
                    ),
                    "gamma_ef_up": _value_with_error(
                        fit.gamma_ef_up, fit.gamma_ef_up_error
                    ),
                }
                for kind, fit in rate_fits.items()
            }
        )
        decay_times["T1_eff"] = {
            kind: _value_with_error(fit.t1_eff, fit.t1_eff_error)
            for kind, fit in rate_fits.items()
        }
        decay_times["T1rho"] = {
            kind: _value_with_error(fit.t1rho, fit.t1rho_error)
            for kind, fit in t1rho_fits.items()
        }
        transition_rates["target_leakage"] = {
            kind: {
                "leakage_rate": _value_with_error(
                    fit.leakage_rate, fit.leakage_rate_error
                ),
                "seepage_rate": _value_with_error(
                    fit.seepage_rate, fit.seepage_rate_error
                ),
                "model": fit.model,
            }
            for kind, fit in leakage_fits.items()
        }

    echo_fits = cast(
        Mapping[str, Mapping[str, ExponentialDecayFit]],
        fits.get(_PHENOMENOLOGICAL_ECHO_FITS, {}),
    )
    for protocol, name in (
        (_CONTROL_T2_ECHO, "T2_echo"),
        (_TARGET_T2RHO_ECHO, "T2rho_echo"),
    ):
        if protocol in echo_fits:
            decay_times[name] = {
                kind: _value_with_error(fit.tau, fit.tau_error)
                for kind, fit in echo_fits[protocol].items()
            }
    if "cr_on_dephasing" in fits:
        dephasing_fits = cast(Mapping[str, CrOnDephasingFit], fits["cr_on_dephasing"])
        dephasing_rates: dict[str, object] = {}
        for protocol, name in (
            (_CONTROL_T2_ECHO, "gamma_phi_control"),
            (_TARGET_T2RHO_ECHO, "gamma_phi_rho_target"),
        ):
            fit = dephasing_fits.get(protocol)
            if fit is not None and fit.success:
                dephasing_rates[name] = _value_with_error(
                    fit.gamma_phi, fit.gamma_phi_error
                )
        transition_rates["cr_on_dephasing"] = dephasing_rates
    return transition_rates, decay_times


def _base_cr_on_noise_from_fits(
    fits: Mapping[str, object],
) -> tuple[CrOnNoise | None, str | None]:
    """Build the measured CR-on dissipation model required by forward fits."""
    required = {"control_rate_model", "target_t1rho", "target_leakage"}
    if not required.issubset(fits):
        return None, "Both control-state population protocols are required."
    rate_fit = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])[
        "actual"
    ]
    t1rho_fit = cast(Mapping[str, TargetT1RhoFit], fits["target_t1rho"])["actual"]
    leakage_fit = cast(Mapping[str, TargetLeakageFit], fits["target_leakage"])["actual"]
    failed = [
        name
        for name, success in (
            ("control rate", rate_fit.success),
            ("target T1rho", t1rho_fit.success),
            ("target leakage", leakage_fit.success),
        )
        if not success
    ]
    if failed:
        return None, f"Required fit failed: {', '.join(failed)}."
    try:
        return (
            CrOnNoise(
                gamma_control_g_to_e=rate_fit.gamma_ge_up,
                gamma_control_e_to_g=rate_fit.gamma_ge_down,
                gamma_control_e_to_f=rate_fit.gamma_ef_up,
                gamma_control_f_to_e=rate_fit.gamma_ef_down,
                target_t1rho=t1rho_fit.t1rho,
                target_leakage_rate=leakage_fit.leakage_rate,
                target_seepage_rate=leakage_fit.seepage_rate,
            ),
            None,
        )
    except (TypeError, ValueError) as exc:
        return None, f"Could not construct the CR-on noise model: {exc}"


def _fixed_zero_unresolved_rates(fits: Mapping[str, object]) -> tuple[str, ...]:
    """Identify rates fixed to zero because the measurement did not resolve them."""
    unresolved: list[str] = []
    if "control_rate_model" in fits:
        control_fit = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])[
            "actual"
        ]
        if control_fit.leakage_model == "outward_only":
            unresolved.append("gamma_control_f_to_e")
        elif control_fit.leakage_model == "inward_only":
            unresolved.append("gamma_control_e_to_f")
        elif control_fit.leakage_model == "none":
            unresolved.extend(("gamma_control_e_to_f", "gamma_control_f_to_e"))
    if "target_leakage" in fits:
        target_fit = cast(Mapping[str, TargetLeakageFit], fits["target_leakage"])[
            "actual"
        ]
        if target_fit.model == "leakage_only":
            unresolved.append("target_seepage_rate")
        elif target_fit.model == "seepage_only":
            unresolved.append("target_leakage_rate")
        elif target_fit.model == "none":
            unresolved.extend(("target_leakage_rate", "target_seepage_rate"))
    return tuple(unresolved)


def _failed_dephasing_fit(
    message: str,
    n_values: ArrayLike,
) -> CrOnDephasingFit:
    """Return a structured failed forward fit."""
    n_array = np.asarray(n_values, dtype=np.int64)
    return CrOnDephasingFit(
        success=False,
        message=message,
        gamma_phi=float("nan"),
        gamma_phi_error=float("nan"),
        covariance=np.full((1, 1), np.nan),
        amplitude=float("nan"),
        offset=float("nan"),
        fitted_values=np.full(n_array.shape, np.nan),
        curve_n_values=n_array,
        curve_values=np.full(n_array.shape, np.nan),
        r_squared=float("nan"),
    )


def analyze_cr_pulse_fidelity(
    gate: PulseSchedule | ZX90GateTiming,
    n_values: ArrayLike,
    control_idle_noise: IdleQubitNoise,
    target_idle_noise: IdleQubitNoise,
    cr_on_noise: CrOnNoise,
    *,
    control_x: ArrayLike | None = None,
    target_z: ArrayLike | None = None,
    control_x_standard_errors: ArrayLike | None = None,
    target_z_standard_errors: ArrayLike | None = None,
    control_x180_duration: float = 0.0,
    target_x180_duration: float = 0.0,
    run_simulation: bool | None = None,
) -> CrPulseFidelityAnalysis:
    """
    Infer CR-on dephasing independently from control-X and target-Z echo data.

    Parameters
    ----------
    gate
        Echoed ZX90 schedule or its extracted timing description.
    n_values
        Repetition indices used by the measured echo-decay curves.
    control_idle_noise, target_idle_noise
        Idle T1 and T2-echo models used outside each CR lobe.
    cr_on_noise
        CR-on dissipative model. Successful fits update only its corresponding
        pure-dephasing field.
    control_x, target_z
        Optional measured primary curves for the control T2-echo and target
        T2rho-echo protocols. At least one is required.
    control_x_standard_errors, target_z_standard_errors
        Optional one-standard-error uncertainties for the supplied curves.
    control_x180_duration, target_x180_duration
        Physical X180 durations in ns used in the control-T2-echo model.
    run_simulation
        ``None`` simulates only when both curves are supplied. ``True``
        requires both curves; ``False`` performs only the supplied independent
        forward fits.

    Returns
    -------
    CrPulseFidelityAnalysis
        Per-curve fit results, the updated CR-on noise model, and an optional
        `CrPulseFidelitySimulationResult`. Fit failures are represented in the
        result rather than replaced by an exponential fallback.
    """
    if control_x is None and target_z is None:
        raise ValueError("At least one of control_x and target_z is required.")
    if control_x is None and control_x_standard_errors is not None:
        raise ValueError("control_x_standard_errors requires control_x.")
    if target_z is None and target_z_standard_errors is not None:
        raise ValueError("target_z_standard_errors requires target_z.")
    if run_simulation is True and (control_x is None or target_z is None):
        raise ValueError("Simulation requires both control_x and target_z.")
    if run_simulation is not None and not isinstance(run_simulation, bool):
        raise TypeError("run_simulation must be a boolean or None.")

    timing = (
        gate if isinstance(gate, ZX90GateTiming) else extract_zx90_gate_timing(gate)
    )
    model = prepare_cr_echo_decay_model(
        timing,
        control_idle_noise,
        target_idle_noise,
        cr_on_noise,
        control_x180_duration,
        target_x180_duration,
    )
    control_fit: CrOnDephasingFit | None = None
    target_fit: CrOnDephasingFit | None = None
    if control_x is not None:
        try:
            control_fit = fit_cr_on_control_dephasing(
                model, n_values, control_x, control_x_standard_errors
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            control_fit = _failed_dephasing_fit(str(exc), n_values)
    if target_z is not None:
        try:
            target_fit = fit_cr_on_target_rotating_frame_dephasing(
                model, n_values, target_z, target_z_standard_errors
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            target_fit = _failed_dephasing_fit(str(exc), n_values)

    fitted_noise = cr_on_noise
    if control_fit is not None and control_fit.success:
        fitted_noise = replace(fitted_noise, gamma_phi_control=control_fit.gamma_phi)
    if target_fit is not None and target_fit.success:
        fitted_noise = replace(fitted_noise, gamma_phi_rho_target=target_fit.gamma_phi)
    failed_fits = [
        (name, fit.message)
        for name, fit in (("control", control_fit), ("target", target_fit))
        if fit is not None and not fit.success
    ]
    simulation_enabled = (
        control_x is not None and target_z is not None
        if run_simulation is None
        else run_simulation
    )
    simulation: CrPulseFidelitySimulationResult | None = None
    simulation_error: str | None = None
    if simulation_enabled and not failed_fits:
        try:
            simulation = simulate_cr_pulse_fidelity(
                timing,
                control_idle_noise,
                target_idle_noise,
                fitted_noise,
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            simulation_error = str(exc)

    if failed_fits:
        message = "CR-on dephasing fit failed: " + "; ".join(
            f"{name}: {fit_message}" for name, fit_message in failed_fits
        )
    elif simulation_error is not None:
        message = f"Fidelity simulation failed: {simulation_error}"
    elif simulation_enabled:
        message = "Independent forward fits and simulation completed."
    else:
        message = "Independent forward fit completed; simulation was not requested."
    return CrPulseFidelityAnalysis(
        success=not failed_fits and simulation_error is None,
        message=message,
        cr_on_noise=fitted_noise,
        control_dephasing_fit=control_fit,
        target_dephasing_fit=target_fit,
        simulation=simulation,
    )


def analyze_cr_pulse_coherence(
    measurements: CrPulseCoherenceMeasurements,
    *,
    relative_uncertainty_threshold: float = 1.0,
    zx90_gate: PulseSchedule | ZX90GateTiming | None = None,
    control_idle_noise: IdleQubitNoise | None = None,
    target_idle_noise: IdleQubitNoise | None = None,
    control_x180_duration: float = 0.0,
    target_x180_duration: float = 0.0,
    run_fidelity_simulation: bool | None = None,
) -> CrPulseCoherenceAnalysis:
    """
    Analyze saved CR-coherence measurements without accessing hardware.

    Parameters
    ----------
    measurements
        Processed actual/reference observables returned by the characterization
        workflow, or an equivalent data set constructed offline.
    relative_uncertainty_threshold
        Maximum relative uncertainty used before an unresolved leakage or
        seepage rate is fixed to zero and a reduced model is refitted.
    zx90_gate
        Echoed ZX90 schedule or timing. Required only for physical echo fits.
    control_idle_noise, target_idle_noise
        Idle-noise inputs required only for physical echo fits.
    control_x180_duration, target_x180_duration
        X180 durations in ns used by the control-T2-echo forward model.
    run_fidelity_simulation
        ``None`` automatically simulates when all four protocols and model
        inputs exist. ``False`` still performs available forward fits.

    Returns
    -------
    CrPulseCoherenceAnalysis
        Actual/reference empirical fits, scalar rate/time summaries, explicit
        fit-status metadata, population-derived base CR-on noise, and optional
        control/target forward-fit and fidelity results.
    """
    threshold = _positive_real(
        relative_uncertainty_threshold,
        default=1.0,
        name="relative_uncertainty_threshold",
    )
    if run_fidelity_simulation is not None and not isinstance(
        run_fidelity_simulation, bool
    ):
        raise TypeError("run_fidelity_simulation must be a boolean or None.")
    _validate_cr_pulse_coherence_measurements(measurements)
    protocols = measurements.protocols
    if run_fidelity_simulation is True and not all(
        protocol in protocols for protocol in (*_GEF_PROTOCOLS, *_PAULI_PROTOCOLS)
    ):
        raise ValueError("Fidelity simulation requires all four protocols.")

    fits = _fit_measured_observables(measurements, threshold)
    base_noise, base_noise_error = _base_cr_on_noise_from_fits(fits)
    selected_pauli = tuple(
        protocol for protocol in _PAULI_PROTOCOLS if protocol in protocols
    )
    model_inputs_ready = all(
        value is not None
        for value in (
            base_noise,
            zx90_gate,
            control_idle_noise,
            target_idle_noise,
        )
    )
    forward_skip_reason = base_noise_error
    if base_noise is not None and not model_inputs_ready:
        forward_skip_reason = (
            "zx90_gate, control_idle_noise, and target_idle_noise are required "
            "for physical forward fitting."
        )

    fidelity_analysis: CrPulseFidelityAnalysis | None = None
    if selected_pauli and model_inputs_ready:
        control_x = (
            measurements.pauli_expectations[_CONTROL_T2_ECHO]["X"]["actual"]
            if _CONTROL_T2_ECHO in selected_pauli
            else None
        )
        target_z = (
            measurements.pauli_expectations[_TARGET_T2RHO_ECHO]["Z"]["actual"]
            if _TARGET_T2RHO_ECHO in selected_pauli
            else None
        )
        control_errors = (
            _finite_errors_or_none(
                measurements.pauli_standard_errors[_CONTROL_T2_ECHO]["X"]["actual"]
            )
            if control_x is not None
            else None
        )
        target_errors = (
            _finite_errors_or_none(
                measurements.pauli_standard_errors[_TARGET_T2RHO_ECHO]["Z"]["actual"]
            )
            if target_z is not None
            else None
        )
        fidelity_analysis = analyze_cr_pulse_fidelity(
            cast(PulseSchedule | ZX90GateTiming, zx90_gate),
            measurements.n_values,
            cast(IdleQubitNoise, control_idle_noise),
            cast(IdleQubitNoise, target_idle_noise),
            cast(CrOnNoise, base_noise),
            control_x=control_x,
            target_z=target_z,
            control_x_standard_errors=control_errors,
            target_z_standard_errors=target_errors,
            control_x180_duration=control_x180_duration,
            target_x180_duration=target_x180_duration,
            run_simulation=run_fidelity_simulation,
        )
        dephasing_fits = {
            protocol: fit
            for protocol, fit in (
                (_CONTROL_T2_ECHO, fidelity_analysis.control_dephasing_fit),
                (_TARGET_T2RHO_ECHO, fidelity_analysis.target_dephasing_fit),
            )
            if fit is not None
        }
        if dephasing_fits:
            fits["cr_on_dephasing"] = dephasing_fits
    elif run_fidelity_simulation is True:
        fidelity_analysis = CrPulseFidelityAnalysis(
            success=False,
            message=forward_skip_reason or "Forward-model inputs are unavailable.",
            cr_on_noise=base_noise,
            control_dephasing_fit=None,
            target_dephasing_fit=None,
            simulation=None,
        )

    dephasing_fits = cast(
        Mapping[str, CrOnDephasingFit], fits.get("cr_on_dephasing", {})
    )
    fixed_zero_rates = _fixed_zero_unresolved_rates(fits)
    fit_status: dict[str, object] = {
        "control_leakage_model": (
            cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])[
                "actual"
            ].leakage_model
            if "control_rate_model" in fits
            else None
        ),
        "target_leakage_model": (
            cast(Mapping[str, TargetLeakageFit], fits["target_leakage"])["actual"].model
            if "target_leakage" in fits
            else None
        ),
        "target_t1rho_model": (
            cast(Mapping[str, TargetT1RhoFit], fits["target_t1rho"])["actual"].model
            if "target_t1rho" in fits
            else None
        ),
        "fixed_zero_unresolved_rates": fixed_zero_rates,
        "forward_fit_success": {
            protocol: bool(
                protocol in dephasing_fits and dephasing_fits[protocol].success
            )
            for protocol in selected_pauli
        },
        "forward_fit_skip_reason": {
            protocol: (
                dephasing_fits[protocol].message
                if protocol in dephasing_fits and not dephasing_fits[protocol].success
                else None
                if protocol in dephasing_fits
                else forward_skip_reason
            )
            for protocol in selected_pauli
        },
    }
    transition_rates, decay_times = _summarize_fit_parameters(fits)
    return CrPulseCoherenceAnalysis(
        fits=fits,
        transition_rates=transition_rates,
        decay_times=decay_times,
        fit_status=fit_status,
        base_cr_on_noise=base_noise,
        fidelity_analysis=fidelity_analysis,
    )


def _error_array(errors: NDArray[np.float64]) -> dict[str, object]:
    """Build a Plotly error-bar configuration without inventing zero errors."""
    return {
        "type": "data",
        "array": errors,
        "visible": bool(np.any(np.isfinite(errors))),
    }


def _add_top_cr_axis(
    figure: go.Figure,
    times_us: NDArray[np.float64],
    cr_counts: tuple[int, ...],
    *,
    axis_name: str = "xaxis2",
) -> None:
    """Add a fixed top axis showing the corresponding ZX90 count."""
    maximum = max(float(times_us[-1]), np.finfo(float).eps)
    figure.update_layout(
        xaxis={"title": "Evolution time (µs)", "range": [0.0, maximum]}
    )
    setattr(
        figure.layout,
        axis_name,
        go.layout.XAxis(
            title="ZX90 schedule count",
            overlaying="x",
            side="top",
            range=[0.0, maximum],
            tickmode="array",
            tickvals=times_us,
            ticktext=[str(value) for value in cr_counts],
            showgrid=False,
        ),
    )


def _rate_curve(
    fit: ThreeLevelRateFit,
    protocol: str,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one fitted GEF population trajectory."""
    if not fit.success:
        return np.full((times.size, len(_STATE_NAMES)), np.nan)
    initial = fit.initial_ground if protocol == _CONTROL_GROUND else fit.initial_excited
    return three_level_population_trajectory(
        times,
        initial,
        (
            fit.gamma_ge_down,
            fit.gamma_ge_up,
            fit.gamma_ef_down,
            fit.gamma_ef_up,
        ),
    )


def _exponential_curve(
    fit: ExponentialDecayFit,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate an offset exponential on a dense time grid."""
    if not fit.success or not np.isfinite(fit.tau) or fit.tau <= 0:
        return np.full(times.shape, np.nan)
    return fit.offset + fit.amplitude * np.exp(-times / fit.tau)


def _target_t1rho_curve(
    fit: TargetT1RhoFit,
    leakage_fit: TargetLeakageFit,
    protocol: str,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one initial-state branch of the common target T1rho fit."""
    if not fit.success or not np.isfinite(fit.t1rho) or fit.t1rho <= 0:
        return np.full(times.shape, np.nan)
    amplitude = (
        fit.amplitude_ground if protocol == _CONTROL_GROUND else fit.amplitude_excited
    )
    correction = np.ones_like(times)
    if fit.model == "leakage_corrected":
        f_population = _target_leakage_curve(leakage_fit, protocol, times)
        if np.any(~np.isfinite(f_population)):
            return np.full(times.shape, np.nan)
        initial_f_population = (
            leakage_fit.initial_ground
            if protocol == _CONTROL_GROUND
            else leakage_fit.initial_excited
        )
        survival = 1 - f_population
        if np.any(survival <= np.finfo(float).eps):
            return np.full(times.shape, np.nan)
        correction = (
            (1 - initial_f_population)
            * np.exp(-fit.leakage_rate_used * times)
            / survival
        )
    return amplitude * correction * np.exp(-times / fit.t1rho)


def _target_leakage_curve(
    fit: TargetLeakageFit,
    protocol: str,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one initial-state branch of the target leakage model."""
    if not fit.success:
        return np.full(times.shape, np.nan)
    initial = fit.initial_ground if protocol == _CONTROL_GROUND else fit.initial_excited
    rate_sum = fit.leakage_rate + fit.seepage_rate
    if rate_sum == 0:
        return np.full(times.shape, initial)
    equilibrium = fit.leakage_rate / rate_sum
    return equilibrium + (initial - equilibrium) * np.exp(-rate_sum * times)


def _control_population_figure(
    measurements: CrPulseCoherenceMeasurements,
    analysis: CrPulseCoherenceAnalysis,
    protocol: str,
) -> go.Figure:
    """Plot measured and fitted control GEF populations."""
    figure = go.Figure()
    times = measurements.times[protocol]
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    fits = cast(Mapping[str, ThreeLevelRateFit], analysis.fits["control_rate_model"])
    for kind in ("reference", "actual"):
        is_reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if is_reference else 1.0
        curves = _rate_curve(fits[kind], protocol, dense_times)
        for index, state in enumerate(_STATE_NAMES):
            color = COLORS[index]
            figure.add_trace(
                go.Scatter(
                    x=times_us,
                    y=measurements.populations[protocol][kind][
                        measurements.control_qubit
                    ][:, index],
                    mode="markers",
                    marker={
                        "color": color,
                        "symbol": "diamond-open" if is_reference else "circle",
                    },
                    opacity=opacity,
                    error_y=_error_array(
                        measurements.population_standard_errors[protocol][kind][
                            measurements.control_qubit
                        ][:, index]
                    ),
                    name=f"{kind} P{state}",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=dense_times * 1e-3,
                    y=curves[:, index],
                    mode="lines",
                    line={"color": color, "dash": "dot" if is_reference else "solid"},
                    opacity=opacity,
                    name=f"{kind} fit P{state}",
                )
            )
    figure.update_layout(
        title=(
            f"{_PROTOCOL_LABELS[protocol]}: control "
            f"{measurements.control_qubit} GEF populations"
        ),
        yaxis={"title": "Population", "range": [0.0, 1.0]},
    )
    _add_top_cr_axis(figure, times_us, measurements.cr_pulse_counts)
    return figure


def _target_figure(
    measurements: CrPulseCoherenceMeasurements,
    analysis: CrPulseCoherenceAnalysis,
    protocol: str,
) -> go.Figure:
    """Plot target GE polarization and F-state leakage."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("GE-normalized polarization", "F-state leakage"),
    )
    times = measurements.times[protocol]
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    t1rho_fits = cast(Mapping[str, TargetT1RhoFit], analysis.fits["target_t1rho"])
    leakage_fits = cast(Mapping[str, TargetLeakageFit], analysis.fits["target_leakage"])
    for kind in ("reference", "actual"):
        is_reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if is_reference else 1.0
        marker = "diamond-open" if is_reference else "circle"
        dash = "dot" if is_reference else "solid"
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=measurements.target_polarizations[protocol][kind],
                mode="markers",
                marker={"color": COLORS[0], "symbol": marker},
                opacity=opacity,
                error_y=_error_array(
                    measurements.target_polarization_standard_errors[protocol][kind]
                ),
                name=f"{kind} polarization",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=dense_times * 1e-3,
                y=_target_t1rho_curve(
                    t1rho_fits[kind],
                    leakage_fits[kind],
                    protocol,
                    dense_times,
                ),
                mode="lines",
                line={"color": COLORS[0], "dash": dash},
                opacity=opacity,
                name=f"{kind} fit",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=measurements.populations[protocol][kind][measurements.target_qubit][
                    :, 2
                ],
                mode="markers",
                marker={"color": COLORS[2], "symbol": marker},
                opacity=opacity,
                error_y=_error_array(
                    measurements.population_standard_errors[protocol][kind][
                        measurements.target_qubit
                    ][:, 2]
                ),
                name=f"{kind} Pf",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=dense_times * 1e-3,
                y=_target_leakage_curve(leakage_fits[kind], protocol, dense_times),
                mode="lines",
                line={"color": COLORS[2], "dash": dash},
                opacity=opacity,
                name=f"{kind} Pf fit",
            ),
            row=2,
            col=1,
        )
    maximum = max(float(times_us[-1]), np.finfo(float).eps)
    figure.update_layout(
        title=f"{_PROTOCOL_LABELS[protocol]}: target {measurements.target_qubit}",
        xaxis3={
            "title": "ZX90 schedule count",
            "overlaying": "x",
            "side": "top",
            "range": [0.0, maximum],
            "tickmode": "array",
            "tickvals": times_us,
            "ticktext": [str(value) for value in measurements.cr_pulse_counts],
            "showgrid": False,
        },
    )
    figure.update_yaxes(title_text="(Pe-Pg)/(Pe+Pg)", range=[-1.05, 1.05], row=1, col=1)
    figure.update_yaxes(title_text="Pf", range=[0.0, 1.0], row=2, col=1)
    figure.update_xaxes(range=[0.0, maximum], row=1, col=1)
    figure.update_xaxes(
        title_text="Evolution time (µs)", range=[0.0, maximum], row=2, col=1
    )
    return figure


def _pauli_figure(
    measurements: CrPulseCoherenceMeasurements,
    analysis: CrPulseCoherenceAnalysis,
    protocol: str,
) -> go.Figure:
    """Plot all measured Pauli components and fit only the primary one."""
    figure = go.Figure()
    times = measurements.times[protocol]
    times_us = times * 1e-3
    dense_times = np.linspace(0.0, float(times[-1]), 500)
    primary = _PRIMARY_PAULI[protocol]
    empirical_fits = cast(
        Mapping[str, Mapping[str, ExponentialDecayFit]],
        analysis.fits[_PHENOMENOLOGICAL_ECHO_FITS],
    )[protocol]
    forward_fits = cast(
        Mapping[str, CrOnDephasingFit], analysis.fits.get("cr_on_dephasing", {})
    )
    forward_fit = forward_fits.get(protocol)
    for kind in ("reference", "actual"):
        is_reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if is_reference else 1.0
        for index, basis in enumerate(measurements.pauli_components[protocol]):
            color = COLORS[index]
            symbol = _PAULI_MARKERS[basis]
            if is_reference:
                symbol = _PAULI_REFERENCE_MARKERS[basis]
            figure.add_trace(
                go.Scatter(
                    x=times_us,
                    y=measurements.pauli_expectations[protocol][basis][kind],
                    mode="markers",
                    marker={"color": color, "symbol": symbol},
                    opacity=opacity,
                    error_y=_error_array(
                        measurements.pauli_standard_errors[protocol][basis][kind]
                    ),
                    name=f"{kind} <{basis}> data",
                )
            )
            if basis != primary:
                continue
            if kind == "actual" and forward_fit is not None and forward_fit.success:
                curve_times = np.interp(
                    forward_fit.curve_n_values,
                    measurements.n_values,
                    times,
                )
                curve = forward_fit.curve_values
                fit_name = "forward fit"
            else:
                curve_times = dense_times
                curve = _exponential_curve(empirical_fits[kind], dense_times)
                fit_name = (
                    "exponential fit" if is_reference else "exponential diagnostic"
                )
            figure.add_trace(
                go.Scatter(
                    x=curve_times * 1e-3,
                    y=curve,
                    mode="lines",
                    line={"color": color, "dash": "dot" if is_reference else "solid"},
                    opacity=opacity,
                    name=f"{kind} <{basis}> {fit_name}",
                )
            )
    figure.update_layout(
        title=(
            f"{_PROTOCOL_LABELS[protocol]}: "
            f"{measurements.control_qubit if protocol == _CONTROL_T2_ECHO else measurements.target_qubit}"
        ),
        yaxis={
            "title": f"<{primary}>"
            if len(measurements.pauli_components[protocol]) == 1
            else "Pauli expectation",
            "range": [-1.05, 1.05],
        },
    )
    _add_top_cr_axis(figure, times_us, measurements.cr_pulse_counts)
    return figure


def plot_cr_pulse_coherence(
    measurements: CrPulseCoherenceMeasurements,
    analysis: CrPulseCoherenceAnalysis,
) -> dict[str, go.Figure]:
    """Create fixed-scale data-and-fit figures for every measured protocol."""
    _validate_cr_pulse_coherence_measurements(measurements)
    figures: dict[str, go.Figure] = {}
    for protocol in _GEF_PROTOCOLS:
        if protocol not in measurements.protocols:
            continue
        figures[f"{protocol}_control_populations"] = _control_population_figure(
            measurements, analysis, protocol
        )
        figures[f"{protocol}_target_polarization"] = _target_figure(
            measurements, analysis, protocol
        )
    for protocol in _PAULI_PROTOCOLS:
        if protocol in measurements.protocols:
            figures[protocol] = _pauli_figure(measurements, analysis, protocol)
    return figures


def print_cr_pulse_fidelity_analysis(
    analysis: CrPulseFidelityAnalysis,
    fixed_zero_unresolved_rates: tuple[str, ...] = (),
) -> None:
    """Print the simulated fidelity limits or the analysis failure reason."""
    simulation = analysis.simulation
    if not analysis.success:
        print(f"CR-pulse fidelity analysis failed: {analysis.message}")
        return
    if simulation is None:
        print(f"CR-pulse fidelity analysis: {analysis.message}")
        return
    print("CR-pulse fidelity simulation:")
    print(
        "  Idle coherence limit:       "
        f"{simulation.idle_coherence_limited_fidelity:.6%}"
    )
    print(
        "  CR-on coherence limit:      "
        f"{simulation.cr_on_coherence_limited_fidelity:.6%}"
    )
    print(
        "  CR-on dissipative limit:    "
        f"{simulation.cr_on_dissipative_limited_fidelity:.6%}"
    )
    print(f"  Average leakage:             {simulation.average_leakage:.6%}")
    if fixed_zero_unresolved_rates:
        print(
            "  Conditional result: unresolved rates were fixed to zero: "
            + ", ".join(fixed_zero_unresolved_rates)
        )


__all__ = [
    "CrOnDephasingFit",
    "CrPulseCoherenceAnalysis",
    "CrPulseCoherenceMeasurements",
    "CrPulseFidelityAnalysis",
    "ExponentialDecayFit",
    "TargetLeakageFit",
    "TargetLeakageModel",
    "TargetT1RhoFit",
    "TargetT1RhoModel",
    "ThreeLevelRateFit",
    "analyze_cr_pulse_coherence",
    "analyze_cr_pulse_fidelity",
    "fit_cr_on_control_dephasing",
    "fit_cr_on_target_rotating_frame_dephasing",
    "fit_exponential_decay",
    "fit_target_leakage",
    "fit_target_t1rho",
    "fit_three_level_rate_model",
    "plot_cr_pulse_coherence",
    "print_cr_pulse_fidelity_analysis",
]
