"""Analyze measured CR-pulse coherence data and infer CR-on noise."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Integral, Real
from typing import Literal, Protocol, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from scipy.linalg import expm
from scipy.optimize import OptimizeResult, brentq, least_squares

from qubex.pulse import PulseSchedule
from qubex.visualization import COLORS

from .cr_pulse_fidelity_simulation import (
    CrOnNoise,
    CrPulseFidelitySimulationResult,
    CrTargetDecayModel,
    IdleQubitNoise,
    ZX90GateTiming,
    extract_zx90_gate_timing,
    prepare_cr_echo_decay_model,
    prepare_cr_target_decay_model,
    simulate_cr_pulse_fidelity,
)

TargetLeakageModel = Literal[
    "leakage_and_seepage",
    "leakage_only",
    "seepage_only",
    "none",
]
TargetT1RhoModel = Literal[
    "exponential",
    "leakage_corrected",
    "control_transition_forward",
]
TargetABLeakageModel = Literal["none", "reduced", "full"]

_RATE_BOUNDARY_FRACTION = 1e-5
# Reject parameter directions whose scaled Jacobian singular value is less than
# this fraction of the best-observed direction. This is deliberately stricter
# than NumPy's machine-precision rank tolerance because a numerically nonzero
# direction can still be experimentally unidentifiable.
_JACOBIAN_IDENTIFIABILITY_RCOND = 1e-7
_JACOBIAN_SENSITIVITY_FLOOR = 1e-8
_MAX_REDUCED_CHI_SQUARED = 4.0
_UNWEIGHTED_RESIDUAL_STEP_FACTOR = 3.0
_MIN_WEIGHTED_DYNAMIC_RANGE = 3.0
_MIN_UNWEIGHTED_RELATIVE_DYNAMIC_RANGE = 1e-3
_MIN_PROFILED_AMPLITUDE_FRACTION = 1e-3
_MIN_WEIGHTED_FORWARD_SENSITIVITY = 1.0
_SINGLE_FORWARD_FIT_PARAMETER_COUNT = 3
_RATE_PARAMETER_COUNT = 4
_EXPONENTIAL_PARAMETER_COUNT = 3
_TARGET_AB_PARAMETER_COUNT = 10
_PROFILE_LIKELIHOOD_COST_DELTA_95 = 1.355
_CONTROL_F_APPROXIMATION_WARNING_THRESHOLD = 0.01
_FIDELITY_PARAMETER_NAMES = (
    "gamma_control_g_to_e",
    "gamma_control_e_to_g",
    "gamma_control_e_to_f",
    "gamma_control_f_to_e",
    "gamma_target_1rho_effective",
    "gamma_target_leakage_effective",
    "gamma_target_seepage_effective",
    "gamma_phi_control",
    "gamma_phi_rho_target",
)
_FIDELITY_OUTPUT_NAMES = (
    "idle_coherence_limited_fidelity",
    "cr_on_coherence_limited_fidelity",
    "cr_on_dissipative_limited_fidelity",
    "average_leakage",
)
_FIDELITY_RELATIVE_DIFFERENCE_STEP = 1e-4
_FIDELITY_UNCERTAINTY_DIFFERENCE_STEP = 1e-2
_FIDELITY_ABSOLUTE_DIFFERENCE_STEP = 1e-10
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
_MINIMUM_LEAKAGE_AXIS_UPPER_LIMIT = 0.01
_LEAKAGE_AXIS_HEADROOM = 1.15
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


@dataclass(frozen=True)
class CrPulseFidelityLinearUncertainty:
    """
    Store local delta-method uncertainty of simulated CR-pulse quantities.

    `parameter_names` defines the rows and columns of `parameter_covariance`
    and the columns of `jacobian`; `output_names` defines the rows and columns
    of `output_covariance` and the rows of `jacobian`.
    """

    success: bool
    message: str
    idle_coherence_limited_fidelity_standard_error: float
    cr_on_coherence_limited_fidelity_standard_error: float
    cr_on_dissipative_limited_fidelity_standard_error: float
    average_leakage_standard_error: float
    output_covariance: NDArray[np.float64]
    parameter_covariance: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    finite_difference_steps: NDArray[np.float64]
    finite_difference_schemes: tuple[str, ...]
    fixed_zero_parameters: tuple[str, ...]
    metadata: Mapping[str, object]
    method: str = "local_delta_method"
    idle_noise_uncertainty_propagated: bool = False


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


def _positive_real(value: float, *, name: str) -> float:
    """Validate and return a positive finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return float(value)


def _jacobian_is_identifiable(
    jacobian: NDArray[np.float64],
    parameter_count: int,
    *,
    rcond: float = _JACOBIAN_IDENTIFIABILITY_RCOND,
    minimum_singular_value: float = _JACOBIAN_SENSITIVITY_FLOOR,
) -> bool:
    """Return whether a scaled Jacobian has enough well-observed directions."""
    if parameter_count < 1:
        raise ValueError("parameter_count must be positive.")
    try:
        singular_values = np.linalg.svd(
            np.asarray(jacobian, dtype=np.float64),
            compute_uv=False,
        )
    except np.linalg.LinAlgError:
        return False
    if singular_values.size < parameter_count:
        return False
    required = singular_values[:parameter_count]
    largest = float(required[0])
    smallest = float(required[-1])
    if (
        not np.all(np.isfinite(required))
        or largest <= minimum_singular_value
        or smallest <= minimum_singular_value
    ):
        return False
    return smallest / largest >= rcond


def _model_is_consistent(
    observed: NDArray[np.float64],
    fitted: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
    *,
    parameter_count: int,
) -> bool:
    """Return whether residuals support using a reduced or zero-rate model."""
    residual = np.asarray(fitted - observed, dtype=np.float64)
    if not np.all(np.isfinite(residual)):
        return False
    if standard_errors is not None:
        degrees_of_freedom = max(residual.size - parameter_count, 1)
        reduced_chi_squared = float(
            np.sum((residual / standard_errors) ** 2) / degrees_of_freedom
        )
        return (
            np.isfinite(reduced_chi_squared)
            and reduced_chi_squared <= _MAX_REDUCED_CHI_SQUARED
        )

    signal_scale = max(float(np.max(np.abs(observed))), 1.0)
    numerical_tolerance = 1e-10 * signal_scale
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    if residual_rms <= numerical_tolerance:
        return True
    if residual.shape[0] < 2:
        return False
    step_rms = float(np.sqrt(np.mean(np.diff(residual, axis=0) ** 2) / 2))
    return (
        np.isfinite(step_rms)
        and step_rms > numerical_tolerance
        and residual_rms <= _UNWEIGHTED_RESIDUAL_STEP_FACTOR * step_rms
    )


def _curve_has_sufficient_dynamic_range(
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
) -> bool:
    """Return whether an observed curve changes by a resolvable amount."""
    dynamic_range = float(np.ptp(values))
    if standard_errors is not None:
        return bool(
            dynamic_range
            >= _MIN_WEIGHTED_DYNAMIC_RANGE * float(np.median(standard_errors))
        )
    signal_scale = max(float(np.max(np.abs(values))), 1.0)
    return bool(dynamic_range >= _MIN_UNWEIGHTED_RELATIVE_DYNAMIC_RANGE * signal_scale)


@dataclass(frozen=True)
class TargetT1RhoFit:
    """Store a single target-polarization T1rho fit."""

    success: bool
    message: str
    t1rho: float
    t1rho_error: float
    amplitude: float
    amplitude_error: float
    covariance: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    r_squared: float
    model: TargetT1RhoModel = "exponential"
    leakage_rate_used: float = 0.0


def _failed_target_t1rho_fit(
    message: str,
    n_values: int,
    *,
    model: TargetT1RhoModel,
    leakage_rate_used: float,
) -> TargetT1RhoFit:
    """Return a structured unresolved target-T1rho fit."""
    return TargetT1RhoFit(
        success=False,
        message=message,
        t1rho=float("nan"),
        t1rho_error=float("nan"),
        amplitude=float("nan"),
        amplitude_error=float("nan"),
        covariance=np.full((2, 2), np.nan),
        fitted_values=np.full(n_values, np.nan),
        r_squared=float("nan"),
        model=model,
        leakage_rate_used=leakage_rate_used,
    )


@dataclass(frozen=True)
class TargetLeakageFit:
    """Store one target leakage/seepage fit and its selected-model parameters."""

    success: bool
    message: str
    model: TargetLeakageModel | None
    leakage_rate: float
    seepage_rate: float
    leakage_rate_error: float
    seepage_rate_error: float
    covariance: NDArray[np.float64]
    initial_population: float
    fitted_values: NDArray[np.float64]
    r_squared: float
    leakage_rate_upper_95: float = float("nan")
    upper_bound_message: str = "not evaluated"


def _failed_target_leakage_fit(
    message: str,
    n_values: int,
) -> TargetLeakageFit:
    """Return a structured unresolved target-leakage fit."""
    return TargetLeakageFit(
        success=False,
        message=message,
        model=None,
        leakage_rate=float("nan"),
        seepage_rate=float("nan"),
        leakage_rate_error=float("nan"),
        seepage_rate_error=float("nan"),
        covariance=np.full((2, 2), np.nan),
        initial_population=float("nan"),
        fitted_values=np.full(n_values, np.nan),
        r_squared=float("nan"),
    )


@dataclass(frozen=True)
class TargetABForwardFit:
    """
    Store the joint control-transition-aware fit of actual A/B target data.

    The covariance parameter order is target 1rho rate for instantaneous
    control G/E, target leakage and seepage for control G, target leakage and
    seepage for control E, initial target-X contrast for A/B, and initial
    target-F population for A/B. Rates use inverse time units of the model.
    """

    success: bool
    message: str
    leakage_model: TargetABLeakageModel | None
    target_t1rho_ground: float
    target_t1rho_excited: float
    target_t1rho_ground_error: float
    target_t1rho_excited_error: float
    leakage_rate_ground: float
    seepage_rate_ground: float
    leakage_rate_excited: float
    seepage_rate_excited: float
    leakage_rate_ground_error: float
    seepage_rate_ground_error: float
    leakage_rate_excited_error: float
    seepage_rate_excited_error: float
    leakage_rate_ground_upper_95: float
    leakage_rate_excited_upper_95: float
    upper_bound_message: str
    initial_target_x_ground: float
    initial_target_x_excited: float
    initial_target_f_ground: float
    initial_target_f_excited: float
    covariance: NDArray[np.float64]
    fitted_target_x: Mapping[str, NDArray[np.float64]]
    fitted_target_f: Mapping[str, NDArray[np.float64]]
    curve_n_values: NDArray[np.int64]
    curve_target_x: Mapping[str, NDArray[np.float64]]
    curve_target_f: Mapping[str, NDArray[np.float64]]
    maximum_control_f_population: float
    r_squared: float


@dataclass(frozen=True)
class _TargetABFitCandidate:
    """Store one identifiable target forward-fit candidate."""

    active_leakage_indices: tuple[int, ...]
    optimization: OptimizeResult
    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64]
    parameter_errors: NDArray[np.float64]
    fitted_target_x: Mapping[str, NDArray[np.float64]]
    fitted_target_f: Mapping[str, NDArray[np.float64]]
    fitted_vector: NDArray[np.float64]


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
    """
    Store a joint ground/excited adjacent-transition three-level rate fit.

    The covariance order is `(gamma_ge_down, gamma_ge_up, gamma_ef_down,
    gamma_ef_up)`. Rows and columns for rates fixed to zero by a reduced model
    contain NaN to distinguish them from active zero-variance directions.
    """

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
    gamma_ef_up_upper_95: float = float("nan")
    upper_bound_message: str = "not evaluated"


@dataclass(frozen=True)
class CrPulseCoherenceMeasurements:
    """
    Measured observables needed to repeat CR-coherence analysis offline.

    Times are in ns, rates inferred from them are in `1/ns`, populations use
    the state order `(g, e, f)`, and every observable contains separate
    `actual` and `reference` arrays. Each protocol's time axis is its
    common actual/reference evolution duration. `cr_pulse_counts` is derived
    as `4 * n_values` rather than stored independently.
    """

    control_qubit: str
    target_qubit: str
    protocols: tuple[str, ...]
    pauli_components: Mapping[str, tuple[str, ...]]
    n_values: tuple[int, ...]
    times: Mapping[str, NDArray[np.float64]]
    populations: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
    population_standard_errors: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ]
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

    @property
    def cr_pulse_counts(self) -> tuple[int, ...]:
        """Return the ZX90 schedule count, `4n`, corresponding to each n value."""
        return tuple(4 * n for n in self.n_values)

    @property
    def target_polarizations(
        self,
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
        """Derive GE-normalized target polarization from the GEF populations."""
        polarizations: dict[str, dict[str, NDArray[np.float64]]] = {}
        for protocol in _GEF_PROTOCOLS:
            if protocol not in self.protocols:
                continue
            polarizations[protocol] = {}
            for kind in ("actual", "reference"):
                population = self.populations[protocol][kind][self.target_qubit]
                denominator = population[:, 0] + population[:, 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    polarizations[protocol][kind] = np.where(
                        denominator > np.finfo(float).eps,
                        (population[:, 1] - population[:, 0]) / denominator,
                        np.nan,
                    )
        return polarizations


@dataclass(frozen=True)
class CrPulseFidelityAnalysis:
    """Store independent dephasing fits, simulation, and optional uncertainty."""

    success: bool
    message: str
    cr_on_noise: CrOnNoise | None
    control_dephasing_fit: CrOnDephasingFit | None
    target_dephasing_fit: CrOnDephasingFit | None
    simulation: CrPulseFidelitySimulationResult | None
    linear_uncertainty: CrPulseFidelityLinearUncertainty | None = None
    conservative_simulation_95: CrPulseFidelitySimulationResult | None = None


@dataclass(frozen=True)
class CrPulseCoherenceAnalysis:
    """Fits and derived quantities obtained from one measurement data set."""

    fits: Mapping[str, object]
    transition_rates: Mapping[str, object]
    decay_times: Mapping[str, object]
    fit_status: Mapping[str, object]
    base_cr_on_noise: CrOnNoise | None
    fidelity_analysis: CrPulseFidelityAnalysis | None

    @property
    def fidelity_uncertainty(self) -> CrPulseFidelityLinearUncertainty | None:
        """Return local delta-method uncertainty when propagation was performed."""
        return (
            None
            if self.fidelity_analysis is None
            else self.fidelity_analysis.linear_uncertainty
        )


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


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    """Return a joint coefficient of determination."""
    residual_sum = float(np.sum((observed - fitted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else float("nan")
    return 1 - residual_sum / total_sum


def fit_target_t1rho(
    times: ArrayLike,
    polarization: ArrayLike,
    standard_errors: ArrayLike | None = None,
    *,
    f_population: ArrayLike | None = None,
    leakage_rate: float = 0.0,
    relative_uncertainty_threshold: float = 0.5,
) -> TargetT1RhoFit:
    """
    Fit one target polarization to an amplitude and T1rho.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    polarization
        Measured `(P_e - P_g) / (P_e + P_g)` values.
    standard_errors
        Optional one-standard-error uncertainties used as absolute weights.
    f_population
        Optional smooth F-population trajectory used to correct leakage and
        seepage dilution.
    leakage_rate
        Computational-to-F rate in inverse `times` units. Defaults to zero.
    relative_uncertainty_threshold
        Maximum retained `sigma(T1rho) / T1rho`. Defaults to 0.5.

    Returns
    -------
    TargetT1RhoFit
        The fitted lifetime in `times` units, amplitude, uncertainties, and
        fitted curve. Unresolved fits contain NaN estimates.

    Notes
    -----
    Without leakage correction the model is
    `A * exp(-time / T1rho)`. With `f_population`, it is
    `A * q(0) / q(t) * exp(-(1 / T1rho + leakage_rate) * time)`, where
    `q(t) = 1 - P_f(t)`. The F-population trajectory and leakage rate are
    treated as fixed; their uncertainties are not propagated into the T1rho
    error.
    """
    time_array, values = _validate_time_series(
        times,
        polarization,
        name="polarization",
        minimum_points=3,
    )
    if values.ndim != 1:
        raise ValueError("polarization must be one-dimensional.")
    errors = _resolve_standard_errors(standard_errors, values.shape)
    _positive_real(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    if isinstance(leakage_rate, bool) or not isinstance(leakage_rate, Real):
        raise TypeError("leakage_rate must be a nonnegative finite real number.")
    if not np.isfinite(leakage_rate) or leakage_rate < 0:
        raise ValueError("leakage_rate must be a nonnegative finite real number.")
    if f_population is None and leakage_rate != 0:
        raise ValueError("f_population is required when leakage_rate is nonzero.")

    model: TargetT1RhoModel = "exponential"
    correction = np.ones_like(time_array)
    if f_population is not None:
        _, pf = _validate_time_series(
            time_array,
            f_population,
            name="f_population",
            minimum_points=3,
        )
        if pf.ndim != 1:
            raise ValueError("f_population must be one-dimensional.")
        if np.any((pf < 0) | (pf >= 1)):
            raise ValueError("F-state populations must lie in [0, 1).")
        survival = 1 - pf
        outward_survival = np.exp(-float(leakage_rate) * time_array)
        correction = survival[0] * outward_survival / survival
        model = "leakage_corrected"
    if not _curve_has_sufficient_dynamic_range(values, errors):
        return _failed_target_t1rho_fit(
            "T1rho is not identifiable because the measured polarization "
            "has insufficient dynamic range.",
            time_array.size,
            model=model,
            leakage_rate_used=float(leakage_rate),
        )
    weights = np.ones_like(values) if errors is None else 1 / errors
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    def curve(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        amplitude, scaled_t1rho = parameters
        return amplitude * correction * np.exp(-scaled_times / scaled_t1rho)

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return (curve(parameters) - values) * weights

    optimization = least_squares(
        residual,
        x0=np.array([values[0], 0.5]),
        bounds=(
            np.array([-np.inf, np.finfo(float).eps]),
            np.array([np.inf, np.inf]),
        ),
        max_nfev=20_000,
    )
    if not optimization.success:
        return _failed_target_t1rho_fit(
            str(optimization.message),
            time_array.size,
            model=model,
            leakage_rate_used=float(leakage_rate),
        )
    fitted_values = curve(optimization.x)
    jacobian = np.asarray(optimization.jac, dtype=np.float64)
    signal_scale = max(
        float(np.ptp(values)),
        float(np.max(np.abs(values))),
        1e-12,
    )
    if abs(float(optimization.x[0])) < _MIN_PROFILED_AMPLITUDE_FRACTION * signal_scale:
        return _failed_target_t1rho_fit(
            "T1rho is not identifiable because the fitted polarization "
            "amplitude is negligible.",
            time_array.size,
            model=model,
            leakage_rate_used=float(leakage_rate),
        )
    if not _jacobian_is_identifiable(jacobian, 2):
        return _failed_target_t1rho_fit(
            "T1rho is not identifiable because the fitted curve has "
            "insufficient parameter sensitivity.",
            time_array.size,
            model=model,
            leakage_rate_used=float(leakage_rate),
        )
    scaled_covariance = _estimate_covariance(
        jacobian,
        float(optimization.cost),
        time_array.size,
        2,
        absolute_weights=errors is not None,
    )
    transform = np.diag([1.0, time_scale])
    covariance = transform @ scaled_covariance @ transform
    parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    t1rho = float(optimization.x[1] * time_scale)
    if (
        not np.all(np.isfinite(parameter_errors))
        or parameter_errors[1] / t1rho >= relative_uncertainty_threshold
    ):
        return _failed_target_t1rho_fit(
            "T1rho is not identifiable because its fitted uncertainty is too large.",
            time_array.size,
            model=model,
            leakage_rate_used=float(leakage_rate),
        )
    return TargetT1RhoFit(
        success=True,
        message=str(optimization.message),
        t1rho=t1rho,
        t1rho_error=float(parameter_errors[1]),
        amplitude=float(optimization.x[0]),
        amplitude_error=float(parameter_errors[0]),
        covariance=covariance,
        fitted_values=np.asarray(fitted_values, dtype=np.float64),
        r_squared=_r_squared(values, fitted_values),
        model=model,
        leakage_rate_used=float(leakage_rate),
    )


def _target_leakage_values(
    times: NDArray[np.float64],
    initial_population: float,
    leakage_rate: float,
    seepage_rate: float,
) -> NDArray[np.float64]:
    """Evaluate one effective computational-to-F rate model."""
    total_rate = leakage_rate + seepage_rate
    if total_rate == 0:
        return np.full(times.shape, initial_population)
    equilibrium = leakage_rate / total_rate
    return equilibrium + (initial_population - equilibrium) * np.exp(
        -total_rate * times
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
    f_population: ArrayLike,
    standard_errors: ArrayLike | None = None,
    *,
    relative_uncertainty_threshold: float = 0.5,
) -> TargetLeakageFit:
    """
    Fit one target F-population curve to leakage and seepage rates.

    Parameters
    ----------
    times
        Strictly increasing times beginning at zero, in any consistent unit.
    f_population
        Measured F-state populations in `[0, 1]`.
    standard_errors
        Optional one-standard-error uncertainties used as absolute weights.
    relative_uncertainty_threshold
        Maximum relative uncertainty retained for a nonzero rate. Defaults to
        0.5.

    Returns
    -------
    TargetLeakageFit
        Leakage and seepage rates in inverse `times` units, reduced-model
        status, uncertainties, and fitted curve. Unresolved dynamics return
        `model=None` with NaN parameters and curve values.

    Notes
    -----
    The model is
    `dPf/dt = leakage_rate * (1-Pf) - seepage_rate * Pf`. The initial F
    population is fitted as a nuisance parameter. Models are considered from
    simplest to most complex: a best-fit constant, each one-rate model, and
    finally the two-rate model. The first model consistent with the measured
    uncertainty and having identifiable free parameters is preferred.
    """
    time_array, values = _validate_time_series(
        times,
        f_population,
        name="f_population",
        minimum_points=3,
    )
    if values.ndim != 1:
        raise ValueError("f_population must be one-dimensional.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Target F-state populations must lie in [0, 1].")
    _positive_real(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    errors = _resolve_standard_errors(standard_errors, values.shape)
    weights = np.ones_like(values) if errors is None else 1 / errors
    time_scale = float(time_array[-1])
    scaled_times = time_array / time_scale

    zero_initial_population = float(np.average(values, weights=weights**2))
    zero_fitted_values = np.full(values.shape, zero_initial_population)

    def model_is_consistent(
        fitted_values: NDArray[np.float64],
        parameter_count: int,
    ) -> bool:
        consistent = _model_is_consistent(
            values,
            fitted_values,
            errors,
            parameter_count=parameter_count,
        )
        if errors is not None:
            return consistent
        residual = np.asarray(fitted_values - values, dtype=np.float64)
        return consistent and not _curve_has_sufficient_dynamic_range(residual, None)

    zero_model_consistent = model_is_consistent(zero_fitted_values, 1)
    if zero_model_consistent:
        return TargetLeakageFit(
            success=True,
            message=(
                "No resolved target leakage or seepage; the fitted constant "
                "zero-rate model is consistent with the data."
            ),
            model="none",
            leakage_rate=0.0,
            seepage_rate=0.0,
            leakage_rate_error=float("nan"),
            seepage_rate_error=float("nan"),
            covariance=np.full((2, 2), np.nan),
            initial_population=zero_initial_population,
            fitted_values=zero_fitted_values,
            r_squared=_r_squared(values, zero_fitted_values),
        )

    def optimize(active_indices: tuple[int, ...]) -> OptimizeResult | None:
        def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            rates = np.zeros(2, dtype=np.float64)
            rates[np.asarray(active_indices)] = parameters[1:]
            fitted_values = _target_leakage_values(
                scaled_times,
                float(parameters[0]),
                float(rates[0]),
                float(rates[1]),
            )
            return (fitted_values - values) * weights

        rate_initial_points = (
            ([0.01, 0.01], [0.1, 0.01], [0.01, 0.1], [0.5, 0.5])
            if len(active_indices) == 2
            else ([0.01], [0.1], [0.5])
        )
        population_initial_points = (zero_initial_population, float(values[0]))
        initial_points = (
            [population, *rates]
            for population in population_initial_points
            for rates in rate_initial_points
        )
        candidates: list[OptimizeResult] = []
        for point in initial_points:
            try:
                candidate = least_squares(
                    residual,
                    x0=np.asarray(point, dtype=np.float64),
                    bounds=(
                        np.zeros(len(active_indices) + 1),
                        np.concatenate(([1.0], np.full(len(active_indices), np.inf))),
                    ),
                    ftol=1e-13,
                    xtol=1e-13,
                    gtol=1e-13,
                    max_nfev=20_000,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            candidates.append(candidate)
        if not candidates:
            return None
        successful = [candidate for candidate in candidates if candidate.success]
        return min(successful or candidates, key=lambda candidate: candidate.cost)

    reduced_candidates: list[
        tuple[
            float,
            TargetLeakageModel,
            int,
            float,
            float,
            float,
            NDArray[np.float64],
            str,
        ]
    ] = []
    reduced_models: tuple[tuple[int, TargetLeakageModel], ...] = (
        (0, "leakage_only"),
        (1, "seepage_only"),
    )
    for rate_index, reduced_model in reduced_models:
        candidate = optimize((rate_index,))
        if candidate is None:
            continue
        candidate_jacobian = np.asarray(candidate.jac, dtype=np.float64)
        candidate_identifiable = _jacobian_is_identifiable(candidate_jacobian, 2)
        candidate_parameter_covariance = (
            _estimate_covariance(
                candidate_jacobian,
                float(candidate.cost),
                time_array.size,
                2,
                absolute_weights=errors is not None,
            )
            if candidate_identifiable
            else np.full((2, 2), np.nan)
        )
        candidate_rate = float(candidate.x[1] / time_scale)
        candidate_error = float(
            np.sqrt(max(candidate_parameter_covariance[1, 1], 0.0)) / time_scale
        )
        candidate_initial_population = float(candidate.x[0])
        candidate_rates = (
            (candidate_rate, 0.0) if rate_index == 0 else (0.0, candidate_rate)
        )
        candidate_fitted_values = _target_leakage_values(
            time_array,
            candidate_initial_population,
            *candidate_rates,
        )
        if (
            candidate.success
            and candidate_identifiable
            and model_is_consistent(candidate_fitted_values, 2)
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
                    candidate_initial_population,
                    candidate_rate,
                    candidate_error,
                    candidate_fitted_values,
                    str(candidate.message),
                )
            )

    if reduced_candidates:
        (
            _,
            model,
            rate_index,
            initial_population,
            selected_rate,
            selected_error,
            fitted_values,
            optimization_message,
        ) = min(reduced_candidates, key=lambda candidate: candidate[0])
        leakage_rate = selected_rate if rate_index == 0 else 0.0
        seepage_rate = selected_rate if rate_index == 1 else 0.0
        covariance = np.full((2, 2), np.nan)
        covariance[rate_index, rate_index] = selected_error**2
        leakage_error = selected_error if rate_index == 0 else float("nan")
        seepage_error = selected_error if rate_index == 1 else float("nan")
        success = True
        fixed_rate = "seepage" if rate_index == 0 else "leakage"
        message = f"{optimization_message} {fixed_rate.capitalize()} fixed to zero."
    else:
        full_optimization = optimize((0, 1))
        if full_optimization is None:
            return _failed_target_leakage_fit(
                "All target leakage least-squares starts failed.",
                time_array.size,
            )
        full_jacobian = np.asarray(full_optimization.jac, dtype=np.float64)
        full_identifiable = _jacobian_is_identifiable(full_jacobian, 3)
        full_parameter_covariance = (
            _estimate_covariance(
                full_jacobian,
                float(full_optimization.cost),
                time_array.size,
                3,
                absolute_weights=errors is not None,
            )
            if full_identifiable
            else np.full((3, 3), np.nan)
        )
        full_rates = np.asarray(full_optimization.x[1:], dtype=np.float64) / time_scale
        full_covariance = full_parameter_covariance[1:, 1:] / time_scale**2
        full_errors = np.sqrt(np.clip(np.diag(full_covariance), 0.0, np.inf))
        full_rate_sum = float(np.sum(full_rates))
        initial_population = float(full_optimization.x[0])
        full_fitted_values = _target_leakage_values(
            time_array,
            initial_population,
            float(full_rates[0]),
            float(full_rates[1]),
        )
        full_model_consistent = model_is_consistent(full_fitted_values, 3)
        rates_resolved = all(
            _parameter_is_resolved(
                float(rate),
                float(error),
                relative_uncertainty_threshold,
                time_scale,
            )
            and rate > _RATE_BOUNDARY_FRACTION * full_rate_sum
            for rate, error in zip(full_rates, full_errors, strict=True)
        )
        if (
            full_optimization.success
            and full_identifiable
            and full_model_consistent
            and rates_resolved
        ):
            model = "leakage_and_seepage"
            leakage_rate, seepage_rate = map(float, full_rates)
            leakage_error, seepage_error = map(float, full_errors)
            covariance = full_covariance
            fitted_values = full_fitted_values
            success = True
            message = str(full_optimization.message)
        else:
            model = None
            leakage_rate = seepage_rate = float("nan")
            leakage_error = seepage_error = float("nan")
            covariance = np.full((2, 2), np.nan)
            initial_population = float("nan")
            fitted_values = np.full(time_array.shape, np.nan)
            success = False
            message = (
                "Nonzero target F-state dynamics are present but leakage "
                "and seepage rates are not identifiable."
            )

    return TargetLeakageFit(
        success=success,
        message=message,
        model=model,
        leakage_rate=leakage_rate,
        seepage_rate=seepage_rate,
        leakage_rate_error=leakage_error,
        seepage_rate_error=seepage_error,
        covariance=covariance,
        initial_population=initial_population,
        fitted_values=np.asarray(fitted_values, dtype=np.float64),
        r_squared=_r_squared(values, fitted_values),
    )


def _failed_target_ab_forward_fit(
    message: str,
    n_values: NDArray[np.int64],
) -> TargetABForwardFit:
    """Return a structured unresolved A/B target forward fit."""
    failed_curve = np.full(n_values.shape, np.nan)
    failed_by_protocol = {protocol: failed_curve.copy() for protocol in _GEF_PROTOCOLS}
    return TargetABForwardFit(
        success=False,
        message=message,
        leakage_model=None,
        target_t1rho_ground=float("nan"),
        target_t1rho_excited=float("nan"),
        target_t1rho_ground_error=float("nan"),
        target_t1rho_excited_error=float("nan"),
        leakage_rate_ground=float("nan"),
        seepage_rate_ground=float("nan"),
        leakage_rate_excited=float("nan"),
        seepage_rate_excited=float("nan"),
        leakage_rate_ground_error=float("nan"),
        seepage_rate_ground_error=float("nan"),
        leakage_rate_excited_error=float("nan"),
        seepage_rate_excited_error=float("nan"),
        leakage_rate_ground_upper_95=float("nan"),
        leakage_rate_excited_upper_95=float("nan"),
        upper_bound_message="Unavailable because the joint forward fit failed.",
        initial_target_x_ground=float("nan"),
        initial_target_x_excited=float("nan"),
        initial_target_f_ground=float("nan"),
        initial_target_f_excited=float("nan"),
        covariance=np.full((_TARGET_AB_PARAMETER_COUNT,) * 2, np.nan),
        fitted_target_x=failed_by_protocol,
        fitted_target_f={protocol: failed_curve.copy() for protocol in _GEF_PROTOCOLS},
        curve_n_values=n_values,
        curve_target_x={protocol: failed_curve.copy() for protocol in _GEF_PROTOCOLS},
        curve_target_f={protocol: failed_curve.copy() for protocol in _GEF_PROTOCOLS},
        maximum_control_f_population=float("nan"),
        r_squared=float("nan"),
    )


def fit_cr_target_decay(
    model: CrTargetDecayModel,
    n_values: ArrayLike,
    initial_control_ground: ArrayLike,
    initial_control_excited: ArrayLike,
    target_x_ground: ArrayLike,
    target_x_excited: ArrayLike,
    target_f_ground: ArrayLike,
    target_f_excited: ArrayLike,
    target_x_ground_errors: ArrayLike | None = None,
    target_x_excited_errors: ArrayLike | None = None,
    target_f_ground_errors: ArrayLike | None = None,
    target_f_excited_errors: ArrayLike | None = None,
    *,
    relative_uncertainty_threshold: float = 0.5,
) -> TargetABForwardFit:
    """
    Jointly fit actual A/B target X and F population with control switching.

    Control transition rates are fixed in `model`. The fitted physical rates
    are target `Gamma_1rho` for control G/E and the active subset of the four
    control-conditioned target leakage/seepage rates. Separate initial target
    X contrast and F population are nuisance parameters for A and B.

    Leakage models are considered from zero through increasing active-rate
    subsets to the full four-rate model. A simpler model is retained whenever
    it is identifiable and consistent with the measurement uncertainty.

    Parameters
    ----------
    model
        Prepared physical model containing the fixed control transition rates
        and full un-echoed ZX90 duration.
    n_values
        Strictly increasing repetition indices beginning at zero. Each index
        represents four applications of the modeled ZX90.
    initial_control_ground, initial_control_excited
        Stage-1 fitted control GEF population vectors for A and B.
    target_x_ground, target_x_excited
        GE-normalized target-X observations for A and B.
    target_f_ground, target_f_excited
        Target F-state populations for A and B.
    target_x_ground_errors, target_x_excited_errors
        Optional one-standard-error uncertainties for the target-X curves.
    target_f_ground_errors, target_f_excited_errors
        Optional one-standard-error uncertainties for the target-F curves.
    relative_uncertainty_threshold
        Maximum retained relative uncertainty of each nonzero fitted rate.

    Returns
    -------
    TargetABForwardFit
        State-conditioned target rates, nuisance parameters, covariance,
        model-selection status, fitted observations, and plotting curves.
    """
    if not isinstance(model, CrTargetDecayModel):
        raise TypeError("model must be a CrTargetDecayModel.")
    n_array = np.asarray(n_values)
    if n_array.ndim != 1 or not np.issubdtype(n_array.dtype, np.integer):
        raise ValueError("n_values must be a one-dimensional integer array.")
    n_array = np.asarray(n_array, dtype=np.int64)
    if n_array.size < 3 or n_array[0] != 0 or np.any(np.diff(n_array) <= 0):
        raise ValueError(
            "n_values must contain at least three values increasing from zero."
        )
    threshold = _positive_real(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    observed_by_protocol = {
        _CONTROL_GROUND: (
            np.asarray(target_x_ground, dtype=np.float64),
            np.asarray(target_f_ground, dtype=np.float64),
        ),
        _CONTROL_EXCITED: (
            np.asarray(target_x_excited, dtype=np.float64),
            np.asarray(target_f_excited, dtype=np.float64),
        ),
    }
    for protocol, (target_x, target_f) in observed_by_protocol.items():
        if target_x.shape != n_array.shape or target_f.shape != n_array.shape:
            raise ValueError(f"{protocol} target arrays must match n_values.")
        if not np.all(np.isfinite(target_x)) or not np.all(np.isfinite(target_f)):
            raise ValueError(f"{protocol} target arrays must be finite.")
        if np.any((target_f < 0) | (target_f > 1)):
            raise ValueError(f"{protocol} target F populations must lie in [0, 1].")
    initial_controls = (
        np.asarray(initial_control_ground, dtype=np.float64),
        np.asarray(initial_control_excited, dtype=np.float64),
    )
    for initial_control in initial_controls:
        if (
            initial_control.shape != (3,)
            or not np.all(np.isfinite(initial_control))
            or np.any(initial_control < 0)
            or not np.isclose(np.sum(initial_control), 1.0)
        ):
            raise ValueError("Initial control populations must be probability vectors.")

    supplied_errors = (
        target_x_ground_errors,
        target_x_excited_errors,
        target_f_ground_errors,
        target_f_excited_errors,
    )
    errors = tuple(
        _resolve_standard_errors(error, n_array.shape) for error in supplied_errors
    )
    weights = tuple(
        np.ones(n_array.shape, dtype=np.float64) if error is None else 1 / error
        for error in errors
    )
    absolute_errors = all(error is not None for error in errors)
    observed_vector = np.concatenate(
        (
            observed_by_protocol[_CONTROL_GROUND][0],
            observed_by_protocol[_CONTROL_EXCITED][0],
            observed_by_protocol[_CONTROL_GROUND][1],
            observed_by_protocol[_CONTROL_EXCITED][1],
        )
    )
    error_vector = (
        np.concatenate(cast(tuple[NDArray[np.float64], ...], errors))
        if absolute_errors
        else None
    )
    rate_scale = model.zx90_duration
    base_full_indices = (0, 1, 6, 7, 8, 9)

    def expand_parameters(
        parameters: NDArray[np.float64],
        active_leakage_indices: tuple[int, ...],
        fixed_scaled_rates: Mapping[int, float],
    ) -> NDArray[np.float64]:
        full = np.zeros(_TARGET_AB_PARAMETER_COUNT, dtype=np.float64)
        full[np.asarray(base_full_indices)] = parameters[:6]
        for parameter_index, rate_index in enumerate(active_leakage_indices, start=6):
            full[2 + rate_index] = parameters[parameter_index]
        for rate_index, value in fixed_scaled_rates.items():
            full[2 + rate_index] = value
        return full

    def predictions(
        full_parameters: NDArray[np.float64],
        counts: NDArray[np.int64],
    ):
        return model.predict_pair(
            counts,
            initial_controls,
            (float(full_parameters[6]), float(full_parameters[7])),
            (float(full_parameters[8]), float(full_parameters[9])),
            (
                float(full_parameters[0] / rate_scale),
                float(full_parameters[1] / rate_scale),
            ),
            (
                float(full_parameters[2] / rate_scale),
                float(full_parameters[4] / rate_scale),
            ),
            (
                float(full_parameters[3] / rate_scale),
                float(full_parameters[5] / rate_scale),
            ),
        )

    def fitted_vector(full_parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        ground, excited = predictions(full_parameters, n_array)
        return np.concatenate(
            (ground.target_x, excited.target_x, ground.target_f, excited.target_f)
        )

    initial_x = tuple(
        float(np.clip(observed_by_protocol[protocol][0][0], -1.0, 1.0))
        for protocol in _GEF_PROTOCOLS
    )
    initial_f = tuple(
        float(np.clip(observed_by_protocol[protocol][1][0], 0.0, 1 - 1e-12))
        for protocol in _GEF_PROTOCOLS
    )

    def optimize(
        active_leakage_indices: tuple[int, ...],
        *,
        fixed_scaled_rates: Mapping[int, float] | None = None,
        initial_parameters: NDArray[np.float64] | None = None,
    ) -> OptimizeResult:
        fixed_rates = {} if fixed_scaled_rates is None else fixed_scaled_rates

        def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            prediction = fitted_vector(
                expand_parameters(parameters, active_leakage_indices, fixed_rates)
            )
            return (prediction - observed_vector) * np.concatenate(weights)

        lower = np.concatenate(
            (
                np.zeros(2),
                np.full(2, -1.0),
                np.zeros(2),
                np.zeros(len(active_leakage_indices)),
            )
        )
        upper = np.concatenate(
            (
                np.full(2, np.inf),
                np.ones(2),
                np.full(2, 1 - 1e-12),
                np.full(len(active_leakage_indices), np.inf),
            )
        )
        if initial_parameters is not None:
            initial_points = (np.clip(initial_parameters, lower, upper),)
        else:
            initial_points = tuple(
                np.asarray(
                    [
                        rate_guess,
                        rate_guess,
                        *initial_x,
                        *initial_f,
                        *([rate_guess] * len(active_leakage_indices)),
                    ],
                    dtype=np.float64,
                )
                for rate_guess in (1e-3, 1e-2)
            )
        candidates: list[OptimizeResult] = []
        for point in initial_points:
            try:
                candidate = least_squares(
                    residual,
                    point,
                    bounds=(lower, upper),
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    max_nfev=400,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError("All A/B target least-squares starts failed.")
        successful = [candidate for candidate in candidates if candidate.success]
        return min(successful or candidates, key=lambda candidate: candidate.cost)

    def evaluate_candidate(
        active_leakage_indices: tuple[int, ...],
    ) -> _TargetABFitCandidate | None:
        try:
            optimization = optimize(active_leakage_indices)
        except (FloatingPointError, RuntimeError, ValueError):
            return None
        jacobian = np.asarray(optimization.jac, dtype=np.float64)
        parameter_count = 6 + len(active_leakage_indices)
        identifiable = _jacobian_is_identifiable(jacobian, parameter_count)
        dynamic_covariance = (
            _estimate_covariance(
                jacobian,
                float(optimization.cost),
                observed_vector.size,
                parameter_count,
                absolute_weights=absolute_errors,
            )
            if identifiable
            else np.full((parameter_count,) * 2, np.nan)
        )
        dynamic_to_full = (
            *base_full_indices,
            *(2 + index for index in active_leakage_indices),
        )
        transform = np.diag(
            [
                1 / rate_scale,
                1 / rate_scale,
                1.0,
                1.0,
                1.0,
                1.0,
                *([1 / rate_scale] * len(active_leakage_indices)),
            ]
        )
        physical_dynamic_covariance = transform @ dynamic_covariance @ transform
        covariance = np.full((_TARGET_AB_PARAMETER_COUNT,) * 2, np.nan)
        covariance[np.ix_(dynamic_to_full, dynamic_to_full)] = (
            physical_dynamic_covariance
        )
        parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
        full_parameters = expand_parameters(optimization.x, active_leakage_indices, {})
        physical_parameters = full_parameters.copy()
        physical_parameters[:6] /= rate_scale
        physical_parameters[6:] = full_parameters[6:]
        fitted = fitted_vector(full_parameters)
        consistent = _model_is_consistent(
            observed_vector,
            fitted,
            error_vector,
            parameter_count=parameter_count,
        )
        if error_vector is None:
            residual = fitted - observed_vector
            consistent = consistent and not _curve_has_sufficient_dynamic_range(
                residual,
                None,
            )
        rate_indices = (0, 1, *(2 + index for index in active_leakage_indices))
        rates_resolved = all(
            _parameter_is_resolved(
                float(physical_parameters[index]),
                float(parameter_errors[index]),
                threshold,
                rate_scale,
            )
            for index in rate_indices
        )
        active_rate_sum = float(np.sum(physical_parameters[list(rate_indices[2:])]))
        active_rates_interior = all(
            physical_parameters[index]
            > _RATE_BOUNDARY_FRACTION * max(active_rate_sum, np.finfo(float).eps)
            for index in rate_indices[2:]
        )
        if not (
            optimization.success
            and identifiable
            and consistent
            and rates_resolved
            and active_rates_interior
        ):
            return None
        ground, excited = predictions(full_parameters, n_array)
        return _TargetABFitCandidate(
            active_leakage_indices=active_leakage_indices,
            optimization=optimization,
            parameters=physical_parameters,
            covariance=covariance,
            parameter_errors=parameter_errors,
            fitted_target_x={
                _CONTROL_GROUND: ground.target_x,
                _CONTROL_EXCITED: excited.target_x,
            },
            fitted_target_f={
                _CONTROL_GROUND: ground.target_f,
                _CONTROL_EXCITED: excited.target_f,
            },
            fitted_vector=fitted,
        )

    selected: _TargetABFitCandidate | None = None
    for active_count in range(5):
        candidates = [
            candidate
            for active_indices in combinations(range(4), active_count)
            if (candidate := evaluate_candidate(active_indices)) is not None
        ]
        if candidates:
            selected = min(
                candidates, key=lambda candidate: candidate.optimization.cost
            )
            break
    if selected is None:
        return _failed_target_ab_forward_fit(
            "A/B target dynamics are present but the physical target rates are not identifiable.",
            n_array,
        )

    upper_bounds = {0: float("nan"), 2: float("nan")}
    upper_bound_messages: list[str] = []
    if not absolute_errors:
        upper_bound_messages.append(
            "95% upper bounds unavailable because absolute measurement "
            "uncertainties were unavailable."
        )
    else:
        baseline_cost = float(selected.optimization.cost)
        for rate_index, rate_name in ((0, "ground leakage"), (2, "excited leakage")):
            if rate_index in selected.active_leakage_indices:
                upper_bound_messages.append(f"{rate_name} is resolved.")
                continue

            def cost_difference(
                scaled_rate: float,
                *,
                profiled_rate_index: int = rate_index,
            ) -> float:
                if scaled_rate == 0:
                    return -_PROFILE_LIKELIHOOD_COST_DELTA_95
                try:
                    profiled = optimize(
                        selected.active_leakage_indices,
                        fixed_scaled_rates={profiled_rate_index: scaled_rate},
                        initial_parameters=np.asarray(
                            selected.optimization.x,
                            dtype=np.float64,
                        ),
                    )
                except (FloatingPointError, RuntimeError, ValueError):
                    return np.inf
                if not profiled.success:
                    return np.inf
                return (
                    float(profiled.cost)
                    - baseline_cost
                    - _PROFILE_LIKELIHOOD_COST_DELTA_95
                )

            upper = 1e-5
            while upper < 100.0 and cost_difference(upper) < 0:
                upper *= 4
            if upper >= 100.0 and cost_difference(upper) < 0:
                upper_bound_messages.append(f"No finite {rate_name} upper crossing.")
                continue
            try:
                crossing = cast(
                    float,
                    brentq(
                        cost_difference,
                        np.float64(0.0),
                        np.float64(upper),
                        xtol=1e-10,
                        rtol=np.float64(1e-6),
                    ),
                )
            except (RuntimeError, ValueError):
                upper_bound_messages.append(f"Could not profile {rate_name}.")
                continue
            upper_bounds[rate_index] = float(crossing / rate_scale)
            upper_bound_messages.append(
                f"{rate_name} has an approximate one-sided 95% "
                "profile-likelihood upper bound."
            )

    dense_n = np.arange(int(n_array[-1]) + 1, dtype=np.int64)
    scaled_parameters = selected.parameters.copy()
    scaled_parameters[:6] *= rate_scale
    scaled_parameters[6:] = selected.parameters[6:]
    dense_ground, dense_excited = predictions(scaled_parameters, dense_n)
    leakage_model: TargetABLeakageModel = (
        "none"
        if not selected.active_leakage_indices
        else "full"
        if len(selected.active_leakage_indices) == 4
        else "reduced"
    )
    maximum_control_f = float(
        max(
            np.max(dense_ground.control_populations[:, 2]),
            np.max(dense_excited.control_populations[:, 2]),
        )
    )
    parameters = selected.parameters
    errors_by_parameter = selected.parameter_errors
    return TargetABForwardFit(
        success=True,
        message=str(selected.optimization.message),
        leakage_model=leakage_model,
        target_t1rho_ground=1 / parameters[0],
        target_t1rho_excited=1 / parameters[1],
        target_t1rho_ground_error=errors_by_parameter[0] / parameters[0] ** 2,
        target_t1rho_excited_error=errors_by_parameter[1] / parameters[1] ** 2,
        leakage_rate_ground=float(parameters[2]),
        seepage_rate_ground=float(parameters[3]),
        leakage_rate_excited=float(parameters[4]),
        seepage_rate_excited=float(parameters[5]),
        leakage_rate_ground_error=float(errors_by_parameter[2]),
        seepage_rate_ground_error=float(errors_by_parameter[3]),
        leakage_rate_excited_error=float(errors_by_parameter[4]),
        seepage_rate_excited_error=float(errors_by_parameter[5]),
        leakage_rate_ground_upper_95=upper_bounds[0],
        leakage_rate_excited_upper_95=upper_bounds[2],
        upper_bound_message=" ".join(upper_bound_messages),
        initial_target_x_ground=float(parameters[6]),
        initial_target_x_excited=float(parameters[7]),
        initial_target_f_ground=float(parameters[8]),
        initial_target_f_excited=float(parameters[9]),
        covariance=selected.covariance,
        fitted_target_x=selected.fitted_target_x,
        fitted_target_f=selected.fitted_target_f,
        curve_n_values=dense_n,
        curve_target_x={
            _CONTROL_GROUND: dense_ground.target_x,
            _CONTROL_EXCITED: dense_excited.target_x,
        },
        curve_target_f={
            _CONTROL_GROUND: dense_ground.target_f,
            _CONTROL_EXCITED: dense_excited.target_f,
        },
        maximum_control_f_population=maximum_control_f,
        r_squared=_r_squared(observed_vector, selected.fitted_vector),
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
    dynamic_range = float(np.ptp(observed))
    if not _curve_has_sufficient_dynamic_range(observed, errors):
        return _failed_dephasing_fit(
            "CR-on dephasing rate is not identifiable because the measured "
            "curve has insufficient dynamic range after affine SPAM profiling.",
            n_array,
        )
    rate_scale = _positive_real(
        cr_lobe_duration,
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
    if not optimization.success:
        return _failed_dephasing_fit(str(optimization.message), n_array)
    profile = evaluate(optimization.x)
    minimum_sensitivity = (
        _MIN_WEIGHTED_FORWARD_SENSITIVITY
        if errors is not None
        else max(
            _JACOBIAN_SENSITIVITY_FLOOR,
            _MIN_PROFILED_AMPLITUDE_FRACTION * dynamic_range,
        )
    )
    jacobian = np.asarray(optimization.jac, dtype=np.float64)
    rate_is_identifiable = _jacobian_is_identifiable(
        jacobian,
        1,
        minimum_singular_value=minimum_sensitivity,
    )
    signal_scale = max(dynamic_range, float(np.max(np.abs(observed))), 1e-12)
    amplitude_is_informative = (
        abs(profile[0]) >= _MIN_PROFILED_AMPLITUDE_FRACTION * signal_scale
    )
    if not amplitude_is_informative:
        return _failed_dephasing_fit(
            "CR-on dephasing rate is not identifiable because the profiled "
            "SPAM amplitude is negligible.",
            n_array,
        )
    if not rate_is_identifiable:
        return _failed_dephasing_fit(
            "CR-on dephasing rate is not identifiable because the fitted "
            "curve has insufficient rate sensitivity after affine SPAM profiling.",
            n_array,
        )
    covariance = (
        _estimate_covariance(
            jacobian,
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
    message = str(optimization.message)
    return CrOnDephasingFit(
        success=True,
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
        Fitted parameters. `tau` uses the same unit as `times`. Curves with
        unresolved dynamic range or an unidentifiable decay-time direction
        return a failed fit with NaN estimates.
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
    if not _curve_has_sufficient_dynamic_range(value_array, errors):
        return _failed_exponential_fit(
            "Exponential decay is not identifiable because the measured "
            "curve has insufficient dynamic range.",
            value_array.size,
        )
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

    if not optimization.success:
        return _failed_exponential_fit(
            str(optimization.message),
            value_array.size,
        )
    amplitude, offset, scaled_tau = optimization.x
    tau = float(scaled_tau * time_scale)
    fitted_values = offset + amplitude * np.exp(-time_array / tau)
    signal_scale = max(
        float(np.ptp(value_array)),
        float(np.max(np.abs(value_array))),
        1e-12,
    )
    if abs(float(amplitude)) < _MIN_PROFILED_AMPLITUDE_FRACTION * signal_scale:
        return _failed_exponential_fit(
            "Exponential decay is not identifiable because the fitted "
            "amplitude is negligible.",
            value_array.size,
        )
    jacobian = np.asarray(optimization.jac, dtype=np.float64)
    if not _jacobian_is_identifiable(
        jacobian,
        _EXPONENTIAL_PARAMETER_COUNT,
    ):
        return _failed_exponential_fit(
            "Exponential decay is not identifiable because the fitted curve "
            "has insufficient parameter sensitivity.",
            value_array.size,
        )
    scaled_covariance = _estimate_covariance(
        jacobian,
        float(optimization.cost),
        value_array.size,
        _EXPONENTIAL_PARAMETER_COUNT,
        absolute_weights=errors is not None,
    )
    transform = np.diag([1.0, 1.0, time_scale])
    covariance = transform @ scaled_covariance @ transform
    errors_by_parameter = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    if not np.all(np.isfinite(errors_by_parameter)):
        return _failed_exponential_fit(
            "Exponential decay is not identifiable because its fitted "
            "uncertainties are not finite.",
            value_array.size,
        )
    return ExponentialDecayFit(
        success=True,
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
    relative_uncertainty_threshold: float = 0.5,
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
        reduced model is refitted. Defaults to 0.5.

    Returns
    -------
    ThreeLevelRateFit
        Directed rates and `T1_eff`. Rates are inverse `times` units. The
        `leakage_model` records whether the full, outward-only, inward-only,
        or no-leakage model was identifiable and consistent with the data.
        Visible E-F dynamics without identifiable rates produce a failed fit.
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
    combined_populations = np.column_stack((population_ground, population_excited))
    combined_errors = (
        np.column_stack((errors_ground, errors_excited))
        if errors_ground is not None and errors_excited is not None
        else None
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
    full_jacobian = np.asarray(full_optimization.jac, dtype=np.float64)
    full_identifiable = _jacobian_is_identifiable(
        full_jacobian,
        _RATE_PARAMETER_COUNT,
    )
    full_scaled_covariance = (
        _estimate_covariance(
            full_jacobian,
            float(full_optimization.cost),
            residual_count,
            _RATE_PARAMETER_COUNT,
            absolute_weights=errors_ground is not None and errors_excited is not None,
        )
        if full_identifiable
        else np.full((_RATE_PARAMETER_COUNT,) * 2, np.nan)
    )
    full_covariance = full_scaled_covariance / time_scale**2
    full_rates = np.asarray(full_optimization.x, dtype=np.float64) / time_scale
    full_errors = np.sqrt(np.clip(np.diag(full_covariance), 0.0, np.inf))
    ef_rate_sum = float(full_rates[2] + full_rates[3])
    full_fitted_ground, full_fitted_excited = trajectories(
        np.asarray(full_optimization.x, dtype=np.float64),
        full_indices,
    )
    full_model_consistent = _model_is_consistent(
        combined_populations,
        np.column_stack((full_fitted_ground, full_fitted_excited)),
        combined_errors,
        parameter_count=_RATE_PARAMETER_COUNT,
    )
    outward_resolved = (
        full_optimization.success
        and full_identifiable
        and full_model_consistent
        and (
            full_optimization.x[3] > 1e-8
            and np.isfinite(full_errors[3])
            and full_errors[3] / full_rates[3] < threshold
            and full_rates[3] > _RATE_BOUNDARY_FRACTION * ef_rate_sum
        )
    )
    seepage_resolved = (
        full_optimization.success
        and full_identifiable
        and full_model_consistent
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
            candidate_jacobian = np.asarray(candidate.jac, dtype=np.float64)
            candidate_identifiable = _jacobian_is_identifiable(
                candidate_jacobian,
                len(candidate_indices),
            )
            candidate_covariance = (
                _estimate_covariance(
                    candidate_jacobian,
                    float(candidate.cost),
                    residual_count,
                    len(candidate_indices),
                    absolute_weights=errors_ground is not None
                    and errors_excited is not None,
                )
                / time_scale**2
                if candidate_identifiable
                else np.full((len(candidate_indices),) * 2, np.nan)
            )
            candidate_rate = float(candidate.x[2] / time_scale)
            candidate_error = float(np.sqrt(max(candidate_covariance[2, 2], 0.0)))
            candidate_ground, candidate_excited = trajectories(
                np.asarray(candidate.x, dtype=np.float64),
                candidate_indices,
            )
            if (
                candidate.success
                and candidate_identifiable
                and _model_is_consistent(
                    combined_populations,
                    np.column_stack((candidate_ground, candidate_excited)),
                    combined_errors,
                    parameter_count=len(candidate_indices),
                )
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
    selected_jacobian = np.asarray(optimization.jac, dtype=np.float64)
    selected_model_identifiable = _jacobian_is_identifiable(
        selected_jacobian,
        len(active_indices),
    )
    active_covariance = (
        _estimate_covariance(
            selected_jacobian,
            float(optimization.cost),
            residual_count,
            len(active_indices),
            absolute_weights=errors_ground is not None and errors_excited is not None,
        )
        / time_scale**2
        if selected_model_identifiable
        else np.full((len(active_indices),) * 2, np.nan)
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
    zero_ef_model_consistent = leakage_model != "none" or _model_is_consistent(
        combined_populations,
        np.column_stack((fitted_ground, fitted_excited)),
        combined_errors,
        parameter_count=len(active_indices),
    )
    returned_leakage_model = leakage_model if zero_ef_model_consistent else None
    fit_message = " ".join(
        part for part in (str(optimization.message), fallback_message) if part
    )
    if not selected_model_identifiable:
        fit_message += (
            " The selected rate model is not identifiable from its scaled Jacobian."
        )
    if not zero_ef_model_consistent:
        fit_message += (
            " E-F dynamics are visible but the leakage and seepage rates are "
            "not identifiable."
        )
    fit_success = bool(
        optimization.success
        and selected_model_identifiable
        and zero_ef_model_consistent
    )
    if not fit_success:
        return _failed_rate_fit(
            fit_message,
            initial_ground,
            initial_excited,
            time_array.size,
        )
    gamma_ef_up_upper_95 = float("nan")
    if 3 in active_indices:
        upper_bound_message = "Control E-to-F leakage is resolved."
    elif errors_ground is None or errors_excited is None:
        upper_bound_message = (
            "95% upper bound unavailable because absolute measurement "
            "uncertainties were unavailable."
        )
    else:
        baseline_cost = float(optimization.cost)

        def profiled_cost_difference(scaled_outward_rate: float) -> float:
            if scaled_outward_rate == 0:
                return -_PROFILE_LIKELIHOOD_COST_DELTA_95

            def residual(free_rates: NDArray[np.float64]) -> NDArray[np.float64]:
                all_rates = np.zeros(_RATE_PARAMETER_COUNT, dtype=np.float64)
                all_rates[np.asarray(active_indices)] = free_rates
                all_rates[3] = scaled_outward_rate
                predicted_ground = three_level_population_trajectory(
                    scaled_times,
                    initial_ground,
                    all_rates,
                )
                predicted_excited = three_level_population_trajectory(
                    scaled_times,
                    initial_excited,
                    all_rates,
                )
                return np.concatenate(
                    (
                        (
                            (predicted_ground - population_ground) * weights_ground
                        ).ravel(),
                        (
                            (predicted_excited - population_excited) * weights_excited
                        ).ravel(),
                    )
                )

            try:
                profiled = least_squares(
                    residual,
                    np.asarray(optimization.x, dtype=np.float64),
                    bounds=(0.0, np.inf),
                    x_scale=1.0,
                    max_nfev=2_000,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                return np.inf
            if not profiled.success:
                return np.inf
            return (
                float(profiled.cost) - baseline_cost - _PROFILE_LIKELIHOOD_COST_DELTA_95
            )

        upper = 1e-5
        while upper < 100.0 and profiled_cost_difference(upper) < 0:
            upper *= 4
        try:
            crossing = cast(
                float,
                brentq(
                    profiled_cost_difference,
                    np.float64(0.0),
                    np.float64(upper),
                    xtol=1e-10,
                    rtol=np.float64(1e-6),
                ),
            )
        except (RuntimeError, ValueError):
            upper_bound_message = "Could not profile control E-to-F leakage."
        else:
            gamma_ef_up_upper_95 = float(crossing / time_scale)
            upper_bound_message = (
                "Control E-to-F leakage has an approximate one-sided 95% "
                "profile-likelihood upper bound."
            )
    return ThreeLevelRateFit(
        success=True,
        message=fit_message,
        leakage_model=returned_leakage_model,
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
        gamma_ef_up_upper_95=gamma_ef_up_upper_95,
        upper_bound_message=upper_bound_message,
    )


def _finite_errors_or_none(
    errors: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Return uncertainty data only when at least one value is usable."""
    return errors if np.any(np.isfinite(errors) & (errors > 0)) else None


def _safe_fit_target_t1rho(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
    *,
    f_population: NDArray[np.float64] | None = None,
    leakage_rate: float = 0.0,
) -> TargetT1RhoFit:
    """Fit target T1rho without discarding data after a numerical failure."""
    try:
        return fit_target_t1rho(
            times,
            values,
            standard_errors,
            f_population=f_population,
            leakage_rate=leakage_rate,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_target_t1rho_fit(
            str(exc),
            times.size,
            model="leakage_corrected" if f_population is not None else "exponential",
            leakage_rate_used=leakage_rate,
        )


def _safe_fit_target_leakage(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    standard_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> TargetLeakageFit:
    """Fit target leakage without discarding data after numerical failure."""
    try:
        return fit_target_leakage(
            times,
            values,
            standard_errors,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_target_leakage_fit(str(exc), times.size)


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


def _safe_fit_cr_target_decay(
    model: CrTargetDecayModel,
    n_values: NDArray[np.int64],
    initial_control_ground: NDArray[np.float64],
    initial_control_excited: NDArray[np.float64],
    target_x_ground: NDArray[np.float64],
    target_x_excited: NDArray[np.float64],
    target_f_ground: NDArray[np.float64],
    target_f_excited: NDArray[np.float64],
    target_x_ground_errors: NDArray[np.float64] | None,
    target_x_excited_errors: NDArray[np.float64] | None,
    target_f_ground_errors: NDArray[np.float64] | None,
    target_f_excited_errors: NDArray[np.float64] | None,
    relative_uncertainty_threshold: float,
) -> TargetABForwardFit:
    """Run the A/B target forward fit and preserve numerical failure."""
    try:
        return fit_cr_target_decay(
            model,
            n_values,
            initial_control_ground,
            initial_control_excited,
            target_x_ground,
            target_x_excited,
            target_f_ground,
            target_f_excited,
            target_x_ground_errors,
            target_x_excited_errors,
            target_f_ground_errors,
            target_f_excited_errors,
            relative_uncertainty_threshold=relative_uncertainty_threshold,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_target_ab_forward_fit(str(exc), n_values)


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
        target_polarizations = measurements.target_polarizations
        control_rate_fits = {
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
        fits["control_rate_model"] = control_rate_fits
        reference_leakage_fits = {
            protocol: {
                "reference": _safe_fit_target_leakage(
                    measurements.times[protocol],
                    populations[protocol]["reference"][target][:, 2],
                    _finite_errors_or_none(errors[protocol]["reference"][target][:, 2]),
                    relative_uncertainty_threshold,
                )
            }
            for protocol in _GEF_PROTOCOLS
        }
        reference_t1rho_fits = {
            protocol: {
                "reference": _safe_fit_target_t1rho(
                    measurements.times[protocol],
                    target_polarizations[protocol]["reference"],
                    _finite_errors_or_none(
                        measurements.target_polarization_standard_errors[protocol][
                            "reference"
                        ]
                    ),
                    relative_uncertainty_threshold,
                    f_population=(
                        reference_leakage_fits[protocol]["reference"].fitted_values
                        if reference_leakage_fits[protocol]["reference"].success
                        and reference_leakage_fits[protocol]["reference"].model
                        != "none"
                        else None
                    ),
                    leakage_rate=(
                        reference_leakage_fits[protocol]["reference"].leakage_rate
                        if reference_leakage_fits[protocol]["reference"].success
                        and reference_leakage_fits[protocol]["reference"].model
                        != "none"
                        else 0.0
                    ),
                )
            }
            for protocol in _GEF_PROTOCOLS
        }
        actual_control_fit = control_rate_fits["actual"]
        n_array = np.asarray(measurements.n_values, dtype=np.int64)
        nonzero = n_array > 0
        per_gate_durations = np.concatenate(
            [
                measurements.times[protocol][nonzero] / (4 * n_array[nonzero])
                for protocol in _GEF_PROTOCOLS
            ]
        )
        if (
            actual_control_fit.success
            and per_gate_durations.size
            and np.all(np.isfinite(per_gate_durations))
            and np.allclose(per_gate_durations, per_gate_durations[0])
        ):
            model = prepare_cr_target_decay_model(
                float(per_gate_durations[0]),
                actual_control_fit.gamma_ge_up,
                actual_control_fit.gamma_ge_down,
                actual_control_fit.gamma_ef_up,
                actual_control_fit.gamma_ef_down,
            )
            target_forward_fit = _safe_fit_cr_target_decay(
                model,
                n_array,
                actual_control_fit.initial_ground,
                actual_control_fit.initial_excited,
                target_polarizations[_CONTROL_GROUND]["actual"],
                target_polarizations[_CONTROL_EXCITED]["actual"],
                populations[_CONTROL_GROUND]["actual"][target][:, 2],
                populations[_CONTROL_EXCITED]["actual"][target][:, 2],
                _finite_errors_or_none(
                    measurements.target_polarization_standard_errors[_CONTROL_GROUND][
                        "actual"
                    ]
                ),
                _finite_errors_or_none(
                    measurements.target_polarization_standard_errors[_CONTROL_EXCITED][
                        "actual"
                    ]
                ),
                _finite_errors_or_none(errors[_CONTROL_GROUND]["actual"][target][:, 2]),
                _finite_errors_or_none(
                    errors[_CONTROL_EXCITED]["actual"][target][:, 2]
                ),
                relative_uncertainty_threshold=relative_uncertainty_threshold,
            )
        else:
            reason = (
                "The actual control-rate fit failed."
                if not actual_control_fit.success
                else "A/B evolution times do not define one un-echoed ZX90 duration."
            )
            target_forward_fit = _failed_target_ab_forward_fit(reason, n_array)
        fits["target_ab_forward"] = target_forward_fit

        def actual_t1rho_fit(protocol: str) -> TargetT1RhoFit:
            ground = protocol == _CONTROL_GROUND
            rate_index = 0 if ground else 1
            amplitude_index = 6 if ground else 7
            t1rho = (
                target_forward_fit.target_t1rho_ground
                if ground
                else target_forward_fit.target_t1rho_excited
            )
            rate = 1 / t1rho
            source_covariance = target_forward_fit.covariance
            covariance = np.array(
                [
                    [
                        source_covariance[amplitude_index, amplitude_index],
                        -source_covariance[amplitude_index, rate_index] / rate**2,
                    ],
                    [
                        -source_covariance[rate_index, amplitude_index] / rate**2,
                        source_covariance[rate_index, rate_index] / rate**4,
                    ],
                ]
            )
            return TargetT1RhoFit(
                success=target_forward_fit.success,
                message=target_forward_fit.message,
                t1rho=t1rho,
                t1rho_error=(
                    target_forward_fit.target_t1rho_ground_error
                    if ground
                    else target_forward_fit.target_t1rho_excited_error
                ),
                amplitude=(
                    target_forward_fit.initial_target_x_ground
                    if ground
                    else target_forward_fit.initial_target_x_excited
                ),
                amplitude_error=float(np.sqrt(max(covariance[0, 0], 0.0))),
                covariance=covariance,
                fitted_values=target_forward_fit.fitted_target_x[protocol],
                r_squared=_r_squared(
                    target_polarizations[protocol]["actual"],
                    target_forward_fit.fitted_target_x[protocol],
                ),
                model="control_transition_forward",
            )

        def actual_leakage_fit(protocol: str) -> TargetLeakageFit:
            ground = protocol == _CONTROL_GROUND
            rate_indices = (2, 3) if ground else (4, 5)
            leakage = (
                target_forward_fit.leakage_rate_ground
                if ground
                else target_forward_fit.leakage_rate_excited
            )
            seepage = (
                target_forward_fit.seepage_rate_ground
                if ground
                else target_forward_fit.seepage_rate_excited
            )
            model_name: TargetLeakageModel = (
                "leakage_and_seepage"
                if leakage > 0 and seepage > 0
                else "leakage_only"
                if leakage > 0
                else "seepage_only"
                if seepage > 0
                else "none"
            )
            return TargetLeakageFit(
                success=target_forward_fit.success,
                message=target_forward_fit.message,
                model=model_name if target_forward_fit.success else None,
                leakage_rate=leakage,
                seepage_rate=seepage,
                leakage_rate_error=(
                    target_forward_fit.leakage_rate_ground_error
                    if ground
                    else target_forward_fit.leakage_rate_excited_error
                ),
                seepage_rate_error=(
                    target_forward_fit.seepage_rate_ground_error
                    if ground
                    else target_forward_fit.seepage_rate_excited_error
                ),
                covariance=target_forward_fit.covariance[
                    np.ix_(rate_indices, rate_indices)
                ],
                initial_population=(
                    target_forward_fit.initial_target_f_ground
                    if ground
                    else target_forward_fit.initial_target_f_excited
                ),
                fitted_values=target_forward_fit.fitted_target_f[protocol],
                r_squared=_r_squared(
                    populations[protocol]["actual"][target][:, 2],
                    target_forward_fit.fitted_target_f[protocol],
                ),
                leakage_rate_upper_95=(
                    target_forward_fit.leakage_rate_ground_upper_95
                    if ground
                    else target_forward_fit.leakage_rate_excited_upper_95
                ),
                upper_bound_message=target_forward_fit.upper_bound_message,
            )

        fits["target_leakage"] = {
            protocol: {
                "actual": actual_leakage_fit(protocol),
                "reference": reference_leakage_fits[protocol]["reference"],
            }
            for protocol in _GEF_PROTOCOLS
        }
        fits["target_t1rho"] = {
            protocol: {
                "actual": actual_t1rho_fit(protocol),
                "reference": reference_t1rho_fits[protocol]["reference"],
            }
            for protocol in _GEF_PROTOCOLS
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


def _effective_target_t1rho(
    ground_fit: TargetT1RhoFit,
    excited_fit: TargetT1RhoFit,
    rate_covariance: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """Return the lifetime associated with the mean of two T1rho rates."""
    effective_rate = 0.5 * (1 / ground_fit.t1rho + 1 / excited_fit.t1rho)
    effective_t1rho = 1 / effective_rate
    if rate_covariance is None:
        rate_variance = 0.25 * (
            (ground_fit.t1rho_error / ground_fit.t1rho**2) ** 2
            + (excited_fit.t1rho_error / excited_fit.t1rho**2) ** 2
        )
    else:
        covariance = np.asarray(rate_covariance, dtype=np.float64)
        if covariance.shape != (2, 2):
            raise ValueError("rate_covariance must have shape (2, 2).")
        rate_variance = 0.25 * float(np.sum(covariance))
    effective_error = np.sqrt(max(rate_variance, 0.0)) / effective_rate**2
    return float(effective_t1rho), float(effective_error)


def _mean_with_error(
    first_value: float,
    first_error: float,
    second_value: float,
    second_error: float,
    covariance: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """Return an arithmetic mean and its propagated standard error."""
    value = 0.5 * (first_value + second_value)
    if covariance is None:
        variance = 0.25 * (first_error**2 + second_error**2)
    else:
        covariance_array = np.asarray(covariance, dtype=np.float64)
        if covariance_array.shape != (2, 2):
            raise ValueError("covariance must have shape (2, 2).")
        # Inactive rates are fixed exactly to zero in the selected model and
        # therefore contribute zero conditional variance. Active-rate entries
        # remain finite in every successful joint fit.
        conditional_covariance = np.where(
            np.isnan(covariance_array),
            0.0,
            covariance_array,
        )
        variance = 0.25 * float(np.sum(conditional_covariance))
    error = np.sqrt(max(variance, 0.0))
    return float(value), float(error)


def _summarize_fit_parameters(
    fits: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build compact rate and decay-time summaries from fit objects."""
    transition_rates: dict[str, object] = {"unit": "1/ns"}
    decay_times: dict[str, object] = {"unit": "ns"}
    if "control_rate_model" in fits:
        rate_fits = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])
        t1rho_fits = cast(
            Mapping[str, Mapping[str, TargetT1RhoFit]], fits["target_t1rho"]
        )
        leakage_fits = cast(
            Mapping[str, Mapping[str, TargetLeakageFit]], fits["target_leakage"]
        )
        target_forward_fit = cast(
            TargetABForwardFit | None,
            fits.get("target_ab_forward"),
        )
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
                    "gamma_ef_up_upper_95": fit.gamma_ef_up_upper_95,
                }
                for kind, fit in rate_fits.items()
            }
        )
        decay_times["T1_eff"] = {
            kind: _value_with_error(fit.t1_eff, fit.t1_eff_error)
            for kind, fit in rate_fits.items()
        }
        t1rho_summary: dict[str, object] = {
            protocol: {
                kind: _value_with_error(fit.t1rho, fit.t1rho_error)
                for kind, fit in fits_by_kind.items()
            }
            for protocol, fits_by_kind in t1rho_fits.items()
        }
        target_leakage_summary: dict[str, object] = {
            protocol: {
                kind: {
                    "leakage_rate": _value_with_error(
                        fit.leakage_rate, fit.leakage_rate_error
                    ),
                    "seepage_rate": _value_with_error(
                        fit.seepage_rate, fit.seepage_rate_error
                    ),
                    "model": fit.model,
                    "leakage_rate_upper_95": fit.leakage_rate_upper_95,
                }
                for kind, fit in fits_by_kind.items()
            }
            for protocol, fits_by_kind in leakage_fits.items()
        }
        for kind in ("actual", "reference"):
            ground_t1rho = t1rho_fits[_CONTROL_GROUND][kind]
            excited_t1rho = t1rho_fits[_CONTROL_EXCITED][kind]
            if ground_t1rho.success and excited_t1rho.success:
                effective_t1rho = _effective_target_t1rho(
                    ground_t1rho,
                    excited_t1rho,
                    (
                        target_forward_fit.covariance[np.ix_((0, 1), (0, 1))]
                        if kind == "actual" and target_forward_fit is not None
                        else None
                    ),
                )
                effective_t1rho_summary = cast(
                    dict[str, object],
                    t1rho_summary.setdefault("effective", {}),
                )
                effective_t1rho_summary[kind] = _value_with_error(*effective_t1rho)

            ground_leakage = leakage_fits[_CONTROL_GROUND][kind]
            excited_leakage = leakage_fits[_CONTROL_EXCITED][kind]
            if ground_leakage.success and excited_leakage.success:
                effective_leakage = _mean_with_error(
                    ground_leakage.leakage_rate,
                    ground_leakage.leakage_rate_error,
                    excited_leakage.leakage_rate,
                    excited_leakage.leakage_rate_error,
                    (
                        target_forward_fit.covariance[np.ix_((2, 4), (2, 4))]
                        if kind == "actual" and target_forward_fit is not None
                        else None
                    ),
                )
                effective_seepage = _mean_with_error(
                    ground_leakage.seepage_rate,
                    ground_leakage.seepage_rate_error,
                    excited_leakage.seepage_rate,
                    excited_leakage.seepage_rate_error,
                    (
                        target_forward_fit.covariance[np.ix_((3, 5), (3, 5))]
                        if kind == "actual" and target_forward_fit is not None
                        else None
                    ),
                )
                effective_leakage_summary = cast(
                    dict[str, object],
                    target_leakage_summary.setdefault("effective", {}),
                )
                effective_leakage_summary[kind] = {
                    "leakage_rate": _value_with_error(*effective_leakage),
                    "seepage_rate": _value_with_error(*effective_seepage),
                }
        decay_times["T1rho"] = t1rho_summary
        transition_rates["target_leakage"] = target_leakage_summary

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
    t1rho_fits = cast(Mapping[str, Mapping[str, TargetT1RhoFit]], fits["target_t1rho"])
    leakage_fits = cast(
        Mapping[str, Mapping[str, TargetLeakageFit]], fits["target_leakage"]
    )
    actual_t1rho_fits = [t1rho_fits[protocol]["actual"] for protocol in _GEF_PROTOCOLS]
    actual_leakage_fits = [
        leakage_fits[protocol]["actual"] for protocol in _GEF_PROTOCOLS
    ]
    failed = [
        name
        for name, success in (
            ("control rate", rate_fit.success),
            *(
                (f"target T1rho ({protocol})", fit.success)
                for protocol, fit in zip(
                    _GEF_PROTOCOLS,
                    actual_t1rho_fits,
                    strict=True,
                )
            ),
            *(
                (f"target leakage ({protocol})", fit.success)
                for protocol, fit in zip(
                    _GEF_PROTOCOLS,
                    actual_leakage_fits,
                    strict=True,
                )
            ),
        )
        if not success
    ]
    if failed:
        return None, f"Required fit failed: {', '.join(failed)}."
    effective_t1rho, _ = _effective_target_t1rho(
        actual_t1rho_fits[0],
        actual_t1rho_fits[1],
    )
    effective_leakage, _ = _mean_with_error(
        actual_leakage_fits[0].leakage_rate,
        actual_leakage_fits[0].leakage_rate_error,
        actual_leakage_fits[1].leakage_rate,
        actual_leakage_fits[1].leakage_rate_error,
    )
    effective_seepage, _ = _mean_with_error(
        actual_leakage_fits[0].seepage_rate,
        actual_leakage_fits[0].seepage_rate_error,
        actual_leakage_fits[1].seepage_rate,
        actual_leakage_fits[1].seepage_rate_error,
    )
    try:
        return (
            CrOnNoise(
                gamma_control_g_to_e=rate_fit.gamma_ge_up,
                gamma_control_e_to_g=rate_fit.gamma_ge_down,
                gamma_control_e_to_f=rate_fit.gamma_ef_up,
                gamma_control_f_to_e=rate_fit.gamma_ef_down,
                target_t1rho=effective_t1rho,
                target_leakage_rate=effective_leakage,
                target_seepage_rate=effective_seepage,
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
        target_fits = cast(
            Mapping[str, Mapping[str, TargetLeakageFit]], fits["target_leakage"]
        )
        for protocol in _GEF_PROTOCOLS:
            target_fit = target_fits[protocol]["actual"]
            suffix = protocol.removeprefix("control_")
            if target_fit.model == "leakage_only":
                unresolved.append(f"target_seepage_rate_{suffix}")
            elif target_fit.model == "seepage_only":
                unresolved.append(f"target_leakage_rate_{suffix}")
            elif target_fit.model == "none":
                unresolved.extend(
                    (
                        f"target_leakage_rate_{suffix}",
                        f"target_seepage_rate_{suffix}",
                    )
                )
    return tuple(unresolved)


def _unresolved_rate_upper_bounds(fits: Mapping[str, object]) -> dict[str, float]:
    """Collect profile-likelihood upper bounds for unresolved outward leakage."""
    bounds: dict[str, float] = {}
    if "control_rate_model" not in fits:
        return bounds
    control_fit = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])[
        "actual"
    ]
    if control_fit.gamma_ef_up == 0 and np.isfinite(control_fit.gamma_ef_up_upper_95):
        bounds["gamma_control_e_to_f"] = control_fit.gamma_ef_up_upper_95
    target_fits = cast(
        Mapping[str, Mapping[str, TargetLeakageFit]],
        fits.get("target_leakage", {}),
    )
    for protocol in _GEF_PROTOCOLS:
        if protocol not in target_fits:
            continue
        fit = target_fits[protocol]["actual"]
        if fit.leakage_rate == 0 and np.isfinite(fit.leakage_rate_upper_95):
            bounds[f"target_leakage_rate_{protocol.removeprefix('control_')}"] = (
                fit.leakage_rate_upper_95
            )
    return bounds


def _conservative_outward_leakage_noise(
    nominal: CrOnNoise,
    fits: Mapping[str, object],
) -> CrOnNoise | None:
    """Apply unresolved outward-rate upper bounds without increasing seepage."""
    bounds = _unresolved_rate_upper_bounds(fits)
    if not bounds:
        return None
    target_ground = bounds.get("target_leakage_rate_ground")
    target_excited = bounds.get("target_leakage_rate_excited")
    target_leakage = nominal.target_leakage_rate
    if target_ground is not None or target_excited is not None:
        leakage_fits = cast(
            Mapping[str, Mapping[str, TargetLeakageFit]],
            fits["target_leakage"],
        )
        target_leakage = 0.5 * (
            (
                leakage_fits[_CONTROL_GROUND]["actual"].leakage_rate
                if target_ground is None
                else target_ground
            )
            + (
                leakage_fits[_CONTROL_EXCITED]["actual"].leakage_rate
                if target_excited is None
                else target_excited
            )
        )
    return replace(
        nominal,
        gamma_control_e_to_f=bounds.get(
            "gamma_control_e_to_f",
            nominal.gamma_control_e_to_f,
        ),
        target_leakage_rate=target_leakage,
    )


def _fidelity_output_vector(
    result: CrPulseFidelitySimulationResult,
) -> NDArray[np.float64]:
    """Return simulator outputs in the delta-method output ordering."""
    return np.asarray(
        [getattr(result, name) for name in _FIDELITY_OUTPUT_NAMES],
        dtype=np.float64,
    )


def _cr_on_noise_parameter_vector(noise: CrOnNoise) -> NDArray[np.float64]:
    """Return one CR-on noise model in the nine-rate propagation basis."""
    target_t1rho_rate = 0.0 if np.isinf(noise.target_t1rho) else 1 / noise.target_t1rho
    return np.asarray(
        (
            noise.gamma_control_g_to_e,
            noise.gamma_control_e_to_g,
            noise.gamma_control_e_to_f,
            noise.gamma_control_f_to_e,
            target_t1rho_rate,
            noise.target_leakage_rate,
            noise.target_seepage_rate,
            noise.gamma_phi_control,
            noise.gamma_phi_rho_target,
        ),
        dtype=np.float64,
    )


def _cr_on_noise_from_parameter_vector(
    template: CrOnNoise,
    parameters: NDArray[np.float64],
) -> CrOnNoise:
    """Replace all stochastic fields of a CR-on noise model from rate values."""
    target_t1rho_rate = float(parameters[4])
    return replace(
        template,
        gamma_control_g_to_e=float(parameters[0]),
        gamma_control_e_to_g=float(parameters[1]),
        gamma_control_e_to_f=float(parameters[2]),
        gamma_control_f_to_e=float(parameters[3]),
        target_t1rho=(
            float("inf") if target_t1rho_rate == 0 else 1 / target_t1rho_rate
        ),
        target_leakage_rate=float(parameters[5]),
        target_seepage_rate=float(parameters[6]),
        gamma_phi_control=float(parameters[7]),
        gamma_phi_rho_target=float(parameters[8]),
    )


def _failed_fidelity_linear_uncertainty(
    message: str,
    *,
    parameter_covariance: NDArray[np.float64] | None = None,
    fixed_zero_parameters: tuple[str, ...] = (),
    covariance_approximation: str = "unavailable",
) -> CrPulseFidelityLinearUncertainty:
    """Return an explicitly unavailable delta-method uncertainty result."""
    covariance = (
        np.full((len(_FIDELITY_PARAMETER_NAMES),) * 2, np.nan)
        if parameter_covariance is None
        else np.asarray(parameter_covariance, dtype=np.float64)
    )
    output_covariance = np.full((len(_FIDELITY_OUTPUT_NAMES),) * 2, np.nan)
    output_covariance[0, :] = 0.0
    output_covariance[:, 0] = 0.0
    jacobian = np.full(
        (len(_FIDELITY_OUTPUT_NAMES), len(_FIDELITY_PARAMETER_NAMES)),
        np.nan,
    )
    jacobian[0, :] = 0.0
    return CrPulseFidelityLinearUncertainty(
        success=False,
        message=message,
        idle_coherence_limited_fidelity_standard_error=0.0,
        cr_on_coherence_limited_fidelity_standard_error=float("nan"),
        cr_on_dissipative_limited_fidelity_standard_error=float("nan"),
        average_leakage_standard_error=float("nan"),
        output_covariance=output_covariance,
        parameter_covariance=covariance,
        jacobian=jacobian,
        parameter_names=_FIDELITY_PARAMETER_NAMES,
        output_names=_FIDELITY_OUTPUT_NAMES,
        finite_difference_steps=np.full(len(_FIDELITY_PARAMETER_NAMES), np.nan),
        finite_difference_schemes=tuple(
            "unavailable" for _ in _FIDELITY_PARAMETER_NAMES
        ),
        fixed_zero_parameters=fixed_zero_parameters,
        metadata={
            "parameter_order": _FIDELITY_PARAMETER_NAMES,
            "output_order": _FIDELITY_OUTPUT_NAMES,
            "covariance_approximation": covariance_approximation,
            "ignored_fit_stage_cross_covariance": (
                covariance_approximation == "block_diagonal_between_fit_stages"
            ),
            "idle_t1_t2_uncertainty_propagated": False,
        },
    )


def propagate_cr_pulse_fidelity_uncertainty(
    gate: PulseSchedule | ZX90GateTiming,
    control_idle_noise: IdleQubitNoise,
    target_idle_noise: IdleQubitNoise,
    cr_on_noise: CrOnNoise,
    parameter_covariance: ArrayLike,
    *,
    nominal_simulation: CrPulseFidelitySimulationResult | None = None,
    fixed_zero_parameters: Sequence[str] = (),
    covariance_approximation: str = "supplied_parameter_covariance",
    relative_difference_step: float = _FIDELITY_RELATIVE_DIFFERENCE_STEP,
    uncertainty_difference_step: float = _FIDELITY_UNCERTAINTY_DIFFERENCE_STEP,
    absolute_difference_step: float = _FIDELITY_ABSOLUTE_DIFFERENCE_STEP,
) -> CrPulseFidelityLinearUncertainty:
    """
    Propagate a local nine-rate covariance through the fidelity simulator.

    The supplied covariance order is given by `parameter_names` in the result.
    Central finite differences are used unless the lower evaluation would have
    a negative rate, in which case a forward difference is used. Parameters
    with exactly zero variance are skipped. Only `simulate_cr_pulse_fidelity`
    is reevaluated; no measurement fit is repeated.

    Idle T1/T2 uncertainty is intentionally excluded, so the idle-fidelity
    variance and its output cross-covariances are returned as exactly zero.

    Parameters
    ----------
    gate
        Echoed or un-echoed ZX90 schedule, or extracted semantic timing.
    control_idle_noise, target_idle_noise
        Nominal idle-noise models. Their uncertainty is not propagated.
    cr_on_noise
        Nominal nine-parameter CR-on noise model.
    parameter_covariance
        Local 9-by-9 rate covariance in the order stored in the returned
        `parameter_names` and `metadata["parameter_order"]`.
    nominal_simulation
        Optional already-computed point estimate. Supplying it avoids one
        redundant simulator evaluation.
    fixed_zero_parameters
        Names of reduced-model parameters conditionally fixed to zero.
    covariance_approximation
        Description of how `parameter_covariance` was assembled. The default
        records that the supplied covariance is propagated as-is. The CR
        coherence workflow identifies its internally assembled covariance as
        block diagonal between independently fitted stages.
    relative_difference_step, uncertainty_difference_step,
    absolute_difference_step
        Factors defining
        `h=max(abs(theta)*relative, sigma*uncertainty, absolute)`.

    Returns
    -------
    CrPulseFidelityLinearUncertainty
        Output and parameter covariance, numerical Jacobian, standard errors,
        finite-difference diagnostics, and approximation metadata.
    """
    covariance = np.asarray(parameter_covariance, dtype=np.float64)
    parameter_count = len(_FIDELITY_PARAMETER_NAMES)
    if covariance.shape != (parameter_count, parameter_count):
        raise ValueError(
            f"parameter_covariance must have shape {(parameter_count, parameter_count)}."
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("parameter_covariance must contain only finite values.")
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-15):
        raise ValueError("parameter_covariance must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    covariance_scale = max(
        float(np.linalg.norm(covariance, ord=2)),
        np.finfo(float).tiny,
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue < -1e-12 * covariance_scale:
        raise ValueError("parameter_covariance must be positive semidefinite.")
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-12 * covariance_scale):
        raise ValueError("parameter_covariance contains a negative variance.")
    for name, value in (
        ("relative_difference_step", relative_difference_step),
        ("uncertainty_difference_step", uncertainty_difference_step),
        ("absolute_difference_step", absolute_difference_step),
    ):
        _positive_real(value, name=name)
    fixed_names = tuple(fixed_zero_parameters)
    unknown_fixed_names = set(fixed_names) - set(_FIDELITY_PARAMETER_NAMES)
    if unknown_fixed_names:
        raise ValueError(
            "fixed_zero_parameters contains unknown names: "
            + ", ".join(sorted(unknown_fixed_names))
        )
    if not isinstance(covariance_approximation, str) or not covariance_approximation:
        raise ValueError("covariance_approximation must be a nonempty string.")

    parameters = _cr_on_noise_parameter_vector(cr_on_noise)
    standard_deviations = np.sqrt(np.clip(diagonal, 0.0, np.inf))
    jacobian = np.zeros(
        (len(_FIDELITY_OUTPUT_NAMES), parameter_count),
        dtype=np.float64,
    )
    steps = np.zeros(parameter_count, dtype=np.float64)
    schemes: list[str] = []
    simulation_count = 0
    if nominal_simulation is None:
        nominal_simulation = simulate_cr_pulse_fidelity(
            gate,
            control_idle_noise,
            target_idle_noise,
            cr_on_noise,
        )
        simulation_count += 1
    nominal_outputs = _fidelity_output_vector(nominal_simulation)

    for index, (parameter, sigma) in enumerate(
        zip(parameters, standard_deviations, strict=True)
    ):
        if sigma == 0:
            schemes.append("skipped_zero_variance")
            continue
        step = max(
            abs(float(parameter)) * float(relative_difference_step),
            float(sigma) * float(uncertainty_difference_step),
            float(absolute_difference_step),
        )
        steps[index] = step
        upper_parameters = parameters.copy()
        upper_parameters[index] += step
        upper = _fidelity_output_vector(
            simulate_cr_pulse_fidelity(
                gate,
                control_idle_noise,
                target_idle_noise,
                _cr_on_noise_from_parameter_vector(
                    cr_on_noise,
                    upper_parameters,
                ),
            )
        )
        simulation_count += 1
        if parameter - step < 0:
            jacobian[:, index] = (upper - nominal_outputs) / step
            schemes.append("forward")
            continue
        lower_parameters = parameters.copy()
        lower_parameters[index] -= step
        lower = _fidelity_output_vector(
            simulate_cr_pulse_fidelity(
                gate,
                control_idle_noise,
                target_idle_noise,
                _cr_on_noise_from_parameter_vector(
                    cr_on_noise,
                    lower_parameters,
                ),
            )
        )
        simulation_count += 1
        jacobian[:, index] = (upper - lower) / (2 * step)
        schemes.append("central")

    # CR-on parameters cannot describe idle T1/T2 uncertainty.
    jacobian[0, :] = 0.0
    output_covariance = jacobian @ covariance @ jacobian.T
    output_covariance = 0.5 * (output_covariance + output_covariance.T)
    output_covariance[0, :] = 0.0
    output_covariance[:, 0] = 0.0
    output_variances = np.diag(output_covariance)
    output_scale = max(
        float(np.linalg.norm(output_covariance, ord=2)),
        np.finfo(float).tiny,
    )
    if np.any(output_variances < -1e-12 * output_scale):
        return _failed_fidelity_linear_uncertainty(
            "Delta-method propagation produced a negative output variance.",
            parameter_covariance=covariance,
            fixed_zero_parameters=fixed_names,
            covariance_approximation=covariance_approximation,
        )
    standard_errors = np.sqrt(np.clip(output_variances, 0.0, np.inf))
    return CrPulseFidelityLinearUncertainty(
        success=True,
        message=(
            "Local covariance propagation completed; independently fitted-stage "
            "cross covariance and idle T1/T2 uncertainty were not propagated."
            if covariance_approximation == "block_diagonal_between_fit_stages"
            else "Local covariance propagation completed; idle T1/T2 uncertainty "
            "was not propagated."
        ),
        idle_coherence_limited_fidelity_standard_error=float(standard_errors[0]),
        cr_on_coherence_limited_fidelity_standard_error=float(standard_errors[1]),
        cr_on_dissipative_limited_fidelity_standard_error=float(standard_errors[2]),
        average_leakage_standard_error=float(standard_errors[3]),
        output_covariance=output_covariance,
        parameter_covariance=covariance,
        jacobian=jacobian,
        parameter_names=_FIDELITY_PARAMETER_NAMES,
        output_names=_FIDELITY_OUTPUT_NAMES,
        finite_difference_steps=steps,
        finite_difference_schemes=tuple(schemes),
        fixed_zero_parameters=fixed_names,
        metadata={
            "parameter_order": _FIDELITY_PARAMETER_NAMES,
            "output_order": _FIDELITY_OUTPUT_NAMES,
            "covariance_approximation": covariance_approximation,
            "ignored_fit_stage_cross_covariance": (
                covariance_approximation == "block_diagonal_between_fit_stages"
            ),
            "idle_t1_t2_uncertainty_propagated": False,
            "finite_difference_step_rule": (
                f"max(abs(theta)*{relative_difference_step:g}, "
                f"sigma*{uncertainty_difference_step:g}, "
                f"{absolute_difference_step:g})"
            ),
            "rate_boundary_rule": "forward_difference_if_theta_minus_h_is_negative",
            "fixed_zero_parameters": fixed_names,
            "additional_fidelity_simulations": simulation_count,
        },
    )


def _conditional_covariance(
    covariance: NDArray[np.float64],
    active_indices: tuple[int, ...],
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    """Set inactive fixed-zero directions to zero after validating active ones."""
    source = np.asarray(covariance, dtype=np.float64)
    if source.shape != (expected_size, expected_size):
        raise ValueError(f"{name} covariance has an unexpected shape {source.shape}.")
    active_covariance = source[np.ix_(active_indices, active_indices)]
    if not np.all(np.isfinite(active_covariance)):
        raise ValueError(f"Active {name} covariance is unavailable or non-finite.")
    conditional = np.zeros_like(source)
    conditional[np.ix_(active_indices, active_indices)] = active_covariance
    return conditional


def _fidelity_parameter_covariance_from_fits(
    fits: Mapping[str, object],
    fidelity_analysis: CrPulseFidelityAnalysis,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """
    Assemble the block-diagonal nine-rate covariance from A/B/C/D fits.

    Control covariance is explicitly reordered from `(e->g, g->e, f->e,
    e->f)`. The first six A/B target-rate directions are projected with the
    three arithmetic-mean rows for effective 1rho, leakage, and seepage.
    Cross-covariance between the four fit stages is intentionally zero.
    """
    if fidelity_analysis.cr_on_noise is None:
        raise ValueError("The fitted CR-on noise model is unavailable.")
    control_fit = cast(Mapping[str, ThreeLevelRateFit], fits["control_rate_model"])[
        "actual"
    ]
    if not control_fit.success or control_fit.leakage_model is None:
        raise ValueError("The actual control-rate covariance is unavailable.")
    control_active_by_model = {
        "full": (0, 1, 2, 3),
        "outward_only": (0, 1, 3),
        "inward_only": (0, 1, 2),
        "none": (0, 1),
    }
    control_active = control_active_by_model[control_fit.leakage_model]
    control_source = _conditional_covariance(
        control_fit.covariance,
        control_active,
        expected_size=4,
        name="control-rate",
    )
    # ThreeLevelRateFit stores (e->g, g->e, f->e, e->f).
    control_reorder = (1, 0, 3, 2)
    control_covariance = control_source[np.ix_(control_reorder, control_reorder)]

    target_fit = cast(TargetABForwardFit, fits["target_ab_forward"])
    if not target_fit.success:
        raise ValueError("The actual A/B target-rate covariance is unavailable.")
    target_rates = np.asarray(
        (
            1 / target_fit.target_t1rho_ground,
            1 / target_fit.target_t1rho_excited,
            target_fit.leakage_rate_ground,
            target_fit.seepage_rate_ground,
            target_fit.leakage_rate_excited,
            target_fit.seepage_rate_excited,
        ),
        dtype=np.float64,
    )
    target_source_covariance = np.asarray(target_fit.covariance, dtype=np.float64)[
        :6, :6
    ]
    target_active = tuple(
        index
        for index, rate in enumerate(target_rates)
        if index < 2 or rate > 0 or np.isfinite(target_source_covariance[index, index])
    )
    target_source = _conditional_covariance(
        target_source_covariance,
        target_active,
        expected_size=6,
        name="A/B target-rate",
    )
    effective_transform = np.asarray(
        (
            (0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.5, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    target_covariance = effective_transform @ target_source @ effective_transform.T

    control_dephasing = fidelity_analysis.control_dephasing_fit
    target_dephasing = fidelity_analysis.target_dephasing_fit
    if control_dephasing is None or not control_dephasing.success:
        raise ValueError("The control-dephasing covariance is unavailable.")
    if target_dephasing is None or not target_dephasing.success:
        raise ValueError("The target-dephasing covariance is unavailable.")
    control_phi_covariance = _conditional_covariance(
        control_dephasing.covariance,
        (0,),
        expected_size=1,
        name="control-dephasing",
    )
    target_phi_covariance = _conditional_covariance(
        target_dephasing.covariance,
        (0,),
        expected_size=1,
        name="target-dephasing",
    )
    control_phi_variance = float(control_phi_covariance[0, 0])
    target_phi_variance = float(target_phi_covariance[0, 0])
    if not np.isfinite(control_phi_variance) or control_phi_variance < 0:
        raise ValueError("The active control-dephasing variance is non-finite.")
    if not np.isfinite(target_phi_variance) or target_phi_variance < 0:
        raise ValueError("The active target-dephasing variance is non-finite.")

    covariance = np.zeros((len(_FIDELITY_PARAMETER_NAMES),) * 2, dtype=np.float64)
    covariance[:4, :4] = control_covariance
    covariance[4:7, 4:7] = target_covariance
    covariance[7, 7] = control_phi_variance
    covariance[8, 8] = target_phi_variance
    fixed_zero_parameters = [
        _FIDELITY_PARAMETER_NAMES[index]
        for index, source_index in enumerate(control_reorder)
        if source_index not in control_active
    ]
    if 2 not in target_active and 4 not in target_active:
        fixed_zero_parameters.append("gamma_target_leakage_effective")
    if 3 not in target_active and 5 not in target_active:
        fixed_zero_parameters.append("gamma_target_seepage_effective")
    return covariance, tuple(fixed_zero_parameters)


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
    control_x180_duration: float | None = None,
    target_x180_duration: float | None = None,
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
        Physical X180 durations in ns used in the control-T2-echo model. Both
        are required when `control_x` is supplied; omitted values are never
        treated as instantaneous pulses.
    run_simulation
        `None` simulates only when both curves are supplied. `True`
        requires both curves; `False` performs only the supplied independent
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
    if control_x is not None and (
        control_x180_duration is None or target_x180_duration is None
    ):
        raise ValueError(
            "control_x180_duration and target_x180_duration are required "
            "when fitting control_x."
        )
    resolved_control_x180_duration = (
        0.0 if control_x180_duration is None else control_x180_duration
    )
    resolved_target_x180_duration = (
        0.0 if target_x180_duration is None else target_x180_duration
    )

    timing = (
        gate if isinstance(gate, ZX90GateTiming) else extract_zx90_gate_timing(gate)
    )
    model = prepare_cr_echo_decay_model(
        timing,
        control_idle_noise,
        target_idle_noise,
        cr_on_noise,
        resolved_control_x180_duration,
        resolved_target_x180_duration,
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
    relative_uncertainty_threshold: float = 0.5,
    zx90_gate: PulseSchedule | ZX90GateTiming | None = None,
    control_idle_noise: IdleQubitNoise | None = None,
    target_idle_noise: IdleQubitNoise | None = None,
    control_x180_duration: float | None = None,
    target_x180_duration: float | None = None,
    run_fidelity_simulation: bool | None = None,
    propagate_fidelity_uncertainty: bool | None = None,
    calculate_conservative_scenario: bool = True,
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
        seepage rate is fixed to zero and a reduced model is refitted. Defaults
        to 0.5.
    zx90_gate
        Echoed ZX90 schedule or timing. Required only for physical echo fits.
    control_idle_noise, target_idle_noise
        Idle-noise inputs required only for physical echo fits.
    control_x180_duration, target_x180_duration
        X180 durations in ns used by the control-T2-echo forward model. Both
        are required when that protocol is physically forward-fitted.
    run_fidelity_simulation
        `None` automatically simulates when all four protocols and model
        inputs exist. `False` still performs available forward fits.
    propagate_fidelity_uncertainty
        Whether to propagate the local block-diagonal covariance of the A/B
        control fit, A/B target fit, C fit, and D fit through the fidelity
        simulator by numerical differentiation. `None` (default) enables it
        automatically only when the nominal fidelity simulation runs; `True`
        explicitly requires it and raises `ValueError` when required inputs
        are unavailable; `False` disables it. Measurement fits are not
        repeated.
    calculate_conservative_scenario
        Whether a successful nominal simulation also evaluates unresolved
        outward leakage at its profile-likelihood upper bounds. C/D
        dephasing rates are refitted under those rates before simulation so
        leakage-induced decay is not counted again as pure dephasing.
        Defaults to `True`.

    Returns
    -------
    CrPulseCoherenceAnalysis
        Actual/reference empirical fits, scalar rate/time summaries, explicit
        fit-status metadata, population-derived base CR-on noise, and optional
        control/target forward-fit, fidelity, and local uncertainty results.

    Notes
    -----
    Actual A/B target X and F data are jointly fit by a control-transition-
    aware two-qutrit model; references retain separate phenomenological fits.
    Per-protocol views of the joint fit are stored as
    `fits[name][protocol][actual_or_reference]`. The corresponding
    control-ground and control-excited values remain separate in the scalar
    summaries; `decay_times["T1rho"]` and
    `transition_rates["target_leakage"]` additionally contain `"effective"`
    entries used to construct `base_cr_on_noise`.
    The control-T2-echo and target-T2rho-echo forward fits are scheduled
    independently. Missing X180 durations therefore skip only the control fit;
    an otherwise valid target fit is still returned, with the control skip
    reason recorded in `fit_status`.
    When requested, fidelity uncertainty uses a block-diagonal approximation
    across the control, target, control-dephasing, and target-dephasing fit
    stages. Within-fit covariance is retained, including the effective target
    rate correlations. Idle T1/T2 uncertainty is not propagated.
    """
    threshold = _positive_real(
        relative_uncertainty_threshold,
        name="relative_uncertainty_threshold",
    )
    if run_fidelity_simulation is not None and not isinstance(
        run_fidelity_simulation, bool
    ):
        raise TypeError("run_fidelity_simulation must be a boolean or None.")
    if propagate_fidelity_uncertainty is not None and not isinstance(
        propagate_fidelity_uncertainty, bool
    ):
        raise TypeError("propagate_fidelity_uncertainty must be a boolean or None.")
    if propagate_fidelity_uncertainty is True and run_fidelity_simulation is False:
        raise ValueError(
            "propagate_fidelity_uncertainty requires fidelity simulation to be enabled."
        )
    if not isinstance(calculate_conservative_scenario, bool):
        raise TypeError("calculate_conservative_scenario must be a boolean.")
    _validate_cr_pulse_coherence_measurements(measurements)
    protocols = measurements.protocols
    all_protocols_available = all(
        protocol in protocols for protocol in (*_GEF_PROTOCOLS, *_PAULI_PROTOCOLS)
    )
    if run_fidelity_simulation is True and not all_protocols_available:
        raise ValueError("Fidelity simulation requires all four protocols.")
    if propagate_fidelity_uncertainty is True and not all_protocols_available:
        raise ValueError(
            "Fidelity uncertainty propagation requires all four protocols."
        )

    fits = _fit_measured_observables(measurements, threshold)
    base_noise, base_noise_error = _base_cr_on_noise_from_fits(fits)
    selected_pauli = tuple(
        protocol for protocol in _PAULI_PROTOCOLS if protocol in protocols
    )
    base_model_inputs_ready = all(
        value is not None
        for value in (
            base_noise,
            zx90_gate,
            control_idle_noise,
            target_idle_noise,
        )
    )
    shared_skip_reason = base_noise_error
    if base_noise is not None and not base_model_inputs_ready:
        shared_skip_reason = (
            "zx90_gate, control_idle_noise, and target_idle_noise are required "
            "for physical forward fitting."
        )
    forward_skip_reasons: dict[str, str | None] = dict.fromkeys(
        selected_pauli, shared_skip_reason
    )
    control_inputs_ready = (
        base_model_inputs_ready and _CONTROL_T2_ECHO in selected_pauli
    )
    if control_inputs_ready and (
        control_x180_duration is None or target_x180_duration is None
    ):
        control_inputs_ready = False
        forward_skip_reasons[_CONTROL_T2_ECHO] = (
            "control_x180_duration and target_x180_duration are required for "
            "the control-T2-echo forward fit."
        )
    target_inputs_ready = (
        base_model_inputs_ready and _TARGET_T2RHO_ECHO in selected_pauli
    )
    forward_skip_message = "; ".join(
        dict.fromkeys(
            reason for reason in forward_skip_reasons.values() if reason is not None
        )
    )

    fidelity_analysis: CrPulseFidelityAnalysis | None = None
    conservative_scenario_message: str | None = None
    conservative_dephasing_refitted = False
    if control_inputs_ready or target_inputs_ready:
        control_x = (
            measurements.pauli_expectations[_CONTROL_T2_ECHO]["X"]["actual"]
            if control_inputs_ready
            else None
        )
        target_z = (
            measurements.pauli_expectations[_TARGET_T2RHO_ECHO]["Z"]["actual"]
            if target_inputs_ready
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
            run_simulation=(
                False
                if run_fidelity_simulation is True
                and (control_x is None or target_z is None)
                else run_fidelity_simulation
            ),
        )
        if calculate_conservative_scenario and fidelity_analysis.simulation is not None:
            conservative_base_noise = _conservative_outward_leakage_noise(
                cast(CrOnNoise, base_noise),
                fits,
            )
            if conservative_base_noise is None:
                conservative_scenario_message = (
                    "No finite unresolved outward-rate upper bounds were available."
                )
            else:
                try:
                    conservative_analysis = analyze_cr_pulse_fidelity(
                        cast(PulseSchedule | ZX90GateTiming, zx90_gate),
                        measurements.n_values,
                        cast(IdleQubitNoise, control_idle_noise),
                        cast(IdleQubitNoise, target_idle_noise),
                        conservative_base_noise,
                        control_x=control_x,
                        target_z=target_z,
                        control_x_standard_errors=control_errors,
                        target_z_standard_errors=target_errors,
                        control_x180_duration=control_x180_duration,
                        target_x180_duration=target_x180_duration,
                        run_simulation=True,
                    )
                except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
                    conservative_simulation = None
                    conservative_scenario_message = str(exc)
                else:
                    conservative_simulation = conservative_analysis.simulation
                    conservative_scenario_message = conservative_analysis.message
                    if conservative_simulation is not None:
                        conservative_dephasing_refitted = True
                        conservative_noise = cast(
                            CrOnNoise,
                            conservative_analysis.cr_on_noise,
                        )
                        conservative_simulation = replace(
                            conservative_simulation,
                            model_metadata={
                                **conservative_simulation.model_metadata,
                                "scenario": (
                                    "individually_unresolved_outward_rates_at_"
                                    "approximate_one_sided_95_percent_upper_bounds"
                                ),
                                "is_confidence_lower_bound": False,
                                "seepage_rates_increased": False,
                                "applied_outward_rate_upper_bounds_95": (
                                    _unresolved_rate_upper_bounds(fits)
                                ),
                                "dephasing_rates_refitted": True,
                                "refitted_gamma_phi_control": (
                                    conservative_noise.gamma_phi_control
                                ),
                                "refitted_gamma_phi_rho_target": (
                                    conservative_noise.gamma_phi_rho_target
                                ),
                            },
                        )
                fidelity_analysis = replace(
                    fidelity_analysis,
                    conservative_simulation_95=conservative_simulation,
                )
        if run_fidelity_simulation is True and (control_x is None or target_z is None):
            missing_message = forward_skip_message or (
                "both control and target forward fits are required."
            )
            prior_failure = (
                f" {fidelity_analysis.message}" if not fidelity_analysis.success else ""
            )
            fidelity_analysis = replace(
                fidelity_analysis,
                success=False,
                message=(
                    "Fidelity simulation was requested but could not run: "
                    f"{missing_message}{prior_failure}"
                ),
                simulation=None,
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
            message=forward_skip_message or "Forward-model inputs are unavailable.",
            cr_on_noise=base_noise,
            control_dephasing_fit=None,
            target_dephasing_fit=None,
            simulation=None,
        )

    uncertainty_mode = (
        "auto"
        if propagate_fidelity_uncertainty is None
        else "explicitly_enabled"
        if propagate_fidelity_uncertainty
        else "disabled"
    )
    uncertainty_enabled = propagate_fidelity_uncertainty is True or (
        propagate_fidelity_uncertainty is None
        and all_protocols_available
        and fidelity_analysis is not None
        and fidelity_analysis.simulation is not None
    )
    if propagate_fidelity_uncertainty is True and (
        fidelity_analysis is None or fidelity_analysis.simulation is None
    ):
        reason = (
            forward_skip_message
            if fidelity_analysis is None
            else fidelity_analysis.message
        )
        raise ValueError(
            "Fidelity uncertainty propagation requires a successful nominal "
            f"fidelity simulation: {reason}"
        )

    if uncertainty_enabled:
        uncertainty: CrPulseFidelityLinearUncertainty
        if fidelity_analysis is None or fidelity_analysis.cr_on_noise is None:
            uncertainty = _failed_fidelity_linear_uncertainty(
                "The fitted CR-on noise model is unavailable."
            )
        elif any(
            value is None
            for value in (zx90_gate, control_idle_noise, target_idle_noise)
        ):
            uncertainty = _failed_fidelity_linear_uncertainty(
                "Fidelity simulator inputs are unavailable."
            )
        else:
            try:
                parameter_covariance, fixed_zero_parameters = (
                    _fidelity_parameter_covariance_from_fits(
                        fits,
                        fidelity_analysis,
                    )
                )
                uncertainty = propagate_cr_pulse_fidelity_uncertainty(
                    cast(PulseSchedule | ZX90GateTiming, zx90_gate),
                    cast(IdleQubitNoise, control_idle_noise),
                    cast(IdleQubitNoise, target_idle_noise),
                    fidelity_analysis.cr_on_noise,
                    parameter_covariance,
                    nominal_simulation=fidelity_analysis.simulation,
                    fixed_zero_parameters=fixed_zero_parameters,
                    covariance_approximation="block_diagonal_between_fit_stages",
                )
            except (
                FloatingPointError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                uncertainty = _failed_fidelity_linear_uncertainty(
                    f"Fidelity covariance propagation is unavailable: {exc}"
                )
        if propagate_fidelity_uncertainty is True and not uncertainty.success:
            raise ValueError(
                "Explicit fidelity uncertainty propagation failed: "
                f"{uncertainty.message}"
            )
        if fidelity_analysis is None:  # pragma: no cover - guarded above
            raise RuntimeError("Uncertainty enabled without a fidelity analysis.")
        fidelity_analysis = replace(
            fidelity_analysis,
            linear_uncertainty=uncertainty,
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
            {
                protocol: fit_by_kind["actual"].model
                for protocol, fit_by_kind in cast(
                    Mapping[str, Mapping[str, TargetLeakageFit]],
                    fits["target_leakage"],
                ).items()
            }
            if "target_leakage" in fits
            else None
        ),
        "target_t1rho_model": (
            {
                protocol: fit_by_kind["actual"].model
                for protocol, fit_by_kind in cast(
                    Mapping[str, Mapping[str, TargetT1RhoFit]],
                    fits["target_t1rho"],
                ).items()
            }
            if "target_t1rho" in fits
            else None
        ),
        "target_effective_rate_aggregation": {
            "target_t1rho": "arithmetic_mean_of_inverse_lifetimes",
            "target_leakage_rate": "arithmetic_mean",
            "target_seepage_rate": "arithmetic_mean",
        },
        "fixed_zero_unresolved_rates": fixed_zero_rates,
        "unresolved_outward_rate_upper_bounds_95": (
            _unresolved_rate_upper_bounds(fits)
        ),
        "unresolved_rate_upper_bound_messages": (
            {
                "control": cast(
                    Mapping[str, ThreeLevelRateFit],
                    fits["control_rate_model"],
                )["actual"].upper_bound_message,
                "target": cast(
                    TargetABForwardFit,
                    fits["target_ab_forward"],
                ).upper_bound_message,
            }
            if "control_rate_model" in fits and "target_ab_forward" in fits
            else {}
        ),
        "upper_bound_interpretation": (
            "Approximate one-sided 95% profile-likelihood limits; the "
            "conservative simulation sets only unresolved outward leakage "
            "rates to these limits, refits C/D dephasing, and is a sensitivity "
            "scenario rather than a confidence lower bound."
        ),
        "positive_negative_cr_lobe_same_noise": True,
        "control_f_target_rate_approximation": "arithmetic_mean_of_g_and_e_rates",
        "control_f_target_rate_approximation_warning_threshold": (
            _CONTROL_F_APPROXIMATION_WARNING_THRESHOLD
        ),
        "maximum_control_f_population_in_target_fit": (
            cast(
                TargetABForwardFit, fits["target_ab_forward"]
            ).maximum_control_f_population
            if "target_ab_forward" in fits
            else None
        ),
        "control_f_target_rate_approximation_material": (
            cast(
                TargetABForwardFit, fits["target_ab_forward"]
            ).maximum_control_f_population
            > _CONTROL_F_APPROXIMATION_WARNING_THRESHOLD
            if "target_ab_forward" in fits
            else False
        ),
        "target_ab_forward_fit_success": (
            cast(TargetABForwardFit, fits["target_ab_forward"]).success
            if "target_ab_forward" in fits
            else False
        ),
        "target_ab_leakage_model": (
            cast(TargetABForwardFit, fits["target_ab_forward"]).leakage_model
            if "target_ab_forward" in fits
            else None
        ),
        "fidelity_uncertainty_requested": propagate_fidelity_uncertainty,
        "fidelity_uncertainty_mode": uncertainty_mode,
        "fidelity_uncertainty_enabled": uncertainty_enabled,
        "fidelity_uncertainty_option_semantics": {
            "None": "auto",
            "True": "explicitly_enabled",
            "False": "disabled",
        },
        "fidelity_uncertainty_method": "local_delta_method",
        "fidelity_uncertainty_parameter_order": _FIDELITY_PARAMETER_NAMES,
        "fidelity_uncertainty_output_order": _FIDELITY_OUTPUT_NAMES,
        "fidelity_uncertainty_covariance_approximation": (
            "block_diagonal_between_fit_stages"
        ),
        "fidelity_uncertainty_success": (
            None
            if not uncertainty_enabled
            else bool(
                fidelity_analysis is not None
                and fidelity_analysis.linear_uncertainty is not None
                and fidelity_analysis.linear_uncertainty.success
            )
        ),
        "fidelity_uncertainty_message": (
            fidelity_analysis.linear_uncertainty.message
            if uncertainty_enabled
            and fidelity_analysis is not None
            and fidelity_analysis.linear_uncertainty is not None
            else "Auto-skipped because nominal fidelity simulation was not executed."
            if uncertainty_mode == "auto"
            else "Disabled by caller."
        ),
        "idle_noise_uncertainty_propagated": False,
        "conservative_outward_scenario": {
            "enabled": calculate_conservative_scenario,
            "dephasing_rates_refitted": conservative_dephasing_refitted,
            "simulation_available": bool(
                fidelity_analysis is not None
                and fidelity_analysis.conservative_simulation_95 is not None
            ),
            "message": conservative_scenario_message,
        },
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
                else forward_skip_reasons[protocol]
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


def _resolve_target_leakage_upper_limit(
    *value_arrays: NDArray[np.float64],
) -> float:
    """Return a padded 1/2/5-series upper limit for finite Pf values."""
    finite_maxima: list[float] = []
    for values in value_arrays:
        array = np.asarray(values, dtype=np.float64)
        finite_values = array[np.isfinite(array)]
        if finite_values.size:
            finite_maxima.append(float(np.max(finite_values)))
    data_max = max(0.0, *finite_maxima)
    padded_max = max(
        _MINIMUM_LEAKAGE_AXIS_UPPER_LIMIT,
        _LEAKAGE_AXIS_HEADROOM * data_max,
    )
    if padded_max >= 1.0:
        return 1.0

    magnitude = 10.0 ** np.floor(np.log10(padded_max))
    normalized = padded_max / magnitude
    multiplier = next(value for value in (1.0, 2.0, 5.0, 10.0) if normalized <= value)
    return min(
        1.0,
        max(_MINIMUM_LEAKAGE_AXIS_UPPER_LIMIT, multiplier * magnitude),
    )


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
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one target T1rho fit on a dense time grid."""
    if not fit.success or not np.isfinite(fit.t1rho) or fit.t1rho <= 0:
        return np.full(times.shape, np.nan)
    correction = np.ones_like(times)
    if fit.model == "leakage_corrected":
        f_population = _target_leakage_curve(leakage_fit, times)
        if np.any(~np.isfinite(f_population)):
            return np.full(times.shape, np.nan)
        survival = 1 - f_population
        if np.any(survival <= np.finfo(float).eps):
            return np.full(times.shape, np.nan)
        correction = (
            (1 - leakage_fit.initial_population)
            * np.exp(-fit.leakage_rate_used * times)
            / survival
        )
    return fit.amplitude * correction * np.exp(-times / fit.t1rho)


def _target_leakage_curve(
    fit: TargetLeakageFit,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one target leakage fit on a dense time grid."""
    if not fit.success:
        return np.full(times.shape, np.nan)
    return _target_leakage_values(
        times,
        fit.initial_population,
        fit.leakage_rate,
        fit.seepage_rate,
    )


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
    t1rho_fits = cast(
        Mapping[str, Mapping[str, TargetT1RhoFit]], analysis.fits["target_t1rho"]
    )[protocol]
    leakage_fits = cast(
        Mapping[str, Mapping[str, TargetLeakageFit]], analysis.fits["target_leakage"]
    )[protocol]
    target_polarizations = measurements.target_polarizations[protocol]
    target_forward_fit = cast(
        TargetABForwardFit | None,
        analysis.fits.get("target_ab_forward"),
    )
    leakage_axis_values: list[NDArray[np.float64]] = []
    for kind in ("reference", "actual"):
        is_reference = kind == "reference"
        opacity = _REFERENCE_OPACITY if is_reference else 1.0
        marker = "diamond-open" if is_reference else "circle"
        dash = "dot" if is_reference else "solid"
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=target_polarizations[kind],
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
        if kind == "actual" and target_forward_fit is not None:
            polarization_curve_times = np.interp(
                target_forward_fit.curve_n_values,
                measurements.n_values,
                times,
            )
            polarization_curve = target_forward_fit.curve_target_x[protocol]
            leakage_curve = target_forward_fit.curve_target_f[protocol]
            leakage_curve_times = polarization_curve_times
        else:
            polarization_curve_times = dense_times
            polarization_curve = _target_t1rho_curve(
                t1rho_fits[kind],
                leakage_fits[kind],
                dense_times,
            )
            leakage_curve_times = dense_times
            leakage_curve = _target_leakage_curve(leakage_fits[kind], dense_times)
        figure.add_trace(
            go.Scatter(
                x=polarization_curve_times * 1e-3,
                y=polarization_curve,
                mode="lines",
                line={"color": COLORS[0], "dash": dash},
                opacity=opacity,
                name=f"{kind} fit",
            ),
            row=1,
            col=1,
        )
        measured_f = measurements.populations[protocol][kind][
            measurements.target_qubit
        ][:, 2]
        measured_f_errors = measurements.population_standard_errors[protocol][kind][
            measurements.target_qubit
        ][:, 2]
        figure.add_trace(
            go.Scatter(
                x=times_us,
                y=measured_f,
                mode="markers",
                marker={"color": COLORS[2], "symbol": marker},
                opacity=opacity,
                error_y=_error_array(measured_f_errors),
                name=f"{kind} Pf",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=leakage_curve_times * 1e-3,
                y=leakage_curve,
                mode="lines",
                line={"color": COLORS[2], "dash": dash},
                opacity=opacity,
                name=f"{kind} Pf fit",
            ),
            row=2,
            col=1,
        )
        leakage_axis_values.extend(
            (
                measured_f,
                measured_f + measured_f_errors,
                np.asarray(leakage_curve, dtype=np.float64),
            )
        )
    leakage_upper_limit = _resolve_target_leakage_upper_limit(*leakage_axis_values)
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
    figure.update_yaxes(
        title_text="Pf",
        range=[0.0, leakage_upper_limit],
        row=2,
        col=1,
    )
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
    """
    Create explicitly ranged figures for measured and fitted coherence diagnostics.

    Parameters
    ----------
    measurements
        Processed actual/reference observables.
    analysis
        Fit results obtained from `analyze_cr_pulse_coherence`.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Figures keyed by protocol and measured subsystem.

    Notes
    -----
    Each target F-population panel independently uses a lower limit of zero
    and an upper limit derived from measured values, their upper one-standard-
    error bars, and actual/reference fit curves. The upper limit includes 15%
    headroom, is rounded upward on a 1/2/5 scale, and is bounded to `[0.01, 1]`.
    """
    _validate_cr_pulse_coherence_measurements(measurements)
    figures: dict[str, go.Figure] = {}
    for protocol in _GEF_PROTOCOLS:
        if protocol not in measurements.protocols:
            continue
        figures[f"{protocol}_control_populations"] = _control_population_figure(
            measurements, analysis, protocol
        )
        figures[f"{protocol}_target_polarization"] = _target_figure(
            measurements,
            analysis,
            protocol,
        )
    for protocol in _PAULI_PROTOCOLS:
        if protocol in measurements.protocols:
            figures[protocol] = _pauli_figure(measurements, analysis, protocol)
    return figures


__all__ = [
    "CrOnDephasingFit",
    "CrPulseCoherenceAnalysis",
    "CrPulseCoherenceMeasurements",
    "CrPulseFidelityAnalysis",
    "CrPulseFidelityLinearUncertainty",
    "ExponentialDecayFit",
    "TargetABForwardFit",
    "TargetABLeakageModel",
    "TargetLeakageFit",
    "TargetLeakageModel",
    "TargetT1RhoFit",
    "TargetT1RhoModel",
    "ThreeLevelRateFit",
    "analyze_cr_pulse_coherence",
    "analyze_cr_pulse_fidelity",
    "fit_cr_on_control_dephasing",
    "fit_cr_on_target_rotating_frame_dephasing",
    "fit_cr_target_decay",
    "fit_exponential_decay",
    "fit_target_leakage",
    "fit_target_t1rho",
    "fit_three_level_rate_model",
    "plot_cr_pulse_coherence",
    "propagate_cr_pulse_fidelity_uncertainty",
]
