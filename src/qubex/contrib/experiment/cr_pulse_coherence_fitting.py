"""Fit measured CR-pulse coherence observables to empirical and physical models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.optimize import least_squares

TargetLeakageModel = Literal["leakage_and_seepage", "leakage_only", "none"]

_RATE_BOUNDARY_FRACTION = 1e-5
_FORWARD_FIT_PARAMETER_COUNT = 6
_RATE_PARAMETER_COUNT = 4
_EXPONENTIAL_PARAMETER_COUNT = 3
_STATE_NAMES = ("g", "e", "f")


class _CrEchoDecayModel(Protocol):
    """Describe the simulation-backed predictor required by the forward fit."""

    @property
    def cr_lobe_duration(self) -> float:
        """Return the CR-lobe duration used to scale fitted rates."""
        ...

    def predict(
        self: _CrEchoDecayModel,
        n_values: NDArray[np.int64],
        gamma_phi_control: float,
        gamma_phi_rho_target: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict control-X and target-Z echo-decay curves."""
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
    """Store the joint forward-fit of CR-on dephasing rates."""

    success: bool
    message: str
    gamma_phi_control: float
    gamma_phi_rho_target: float
    gamma_phi_control_error: float
    gamma_phi_rho_target_error: float
    covariance: NDArray[np.float64]
    control_amplitude: float
    control_offset: float
    target_amplitude: float
    target_offset: float
    fitted_control_x: NDArray[np.float64]
    fitted_target_z: NDArray[np.float64]
    curve_n_values: NDArray[np.int64]
    curve_control_x: NDArray[np.float64]
    curve_target_z: NDArray[np.float64]
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
    leakage_model: Literal["full", "outward_only", "none"] | None
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
    relative_uncertainty_threshold: float = 1.0,
) -> TargetT1RhoFit:
    """
    Jointly fit two polarizations to separate amplitudes and one T1rho.

    The fitted model is ``A * exp(-time / T1rho)``. The two control-state
    protocols have independent amplitudes but share T1rho and a zero
    long-time polarization.
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
        return amplitude_ground * decay, amplitude_excited * decay

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
    identified, seepage is fixed to zero; if leakage is also unresolved, the
    method returns a constant-curve model with both rates fixed to zero.
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

    def optimize(active_parameters: int):
        def residual(scaled_rates: NDArray[np.float64]) -> NDArray[np.float64]:
            leakage = float(scaled_rates[0])
            seepage = float(scaled_rates[1]) if active_parameters == 2 else 0.0
            fitted_ground, fitted_excited = _target_leakage_curves(
                scaled_times,
                initial_ground,
                initial_excited,
                leakage,
                seepage,
            )
            return np.concatenate(
                (
                    (fitted_ground - ground) * weights_ground,
                    (fitted_excited - excited) * weights_excited,
                )
            )

        initial_points = (
            ([0.01, 0.01], [0.1, 0.01], [0.01, 0.1], [0.5, 0.5])
            if active_parameters == 2
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

    full_optimization = optimize(2)
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
        optimization = optimize(1)
        one_covariance = (
            _estimate_covariance(
                np.asarray(optimization.jac, dtype=np.float64),
                float(optimization.cost),
                2 * time_array.size,
                1,
                absolute_weights=errors_ground is not None
                or errors_excited is not None,
            )
            / time_scale**2
        )
        one_rate = float(optimization.x[0] / time_scale)
        one_error = float(np.sqrt(max(one_covariance[0, 0], 0.0)))
        leakage_only_resolved = (
            optimization.success
            and np.linalg.matrix_rank(optimization.jac) == 1
            and _parameter_is_resolved(
                one_rate,
                one_error,
                relative_uncertainty_threshold,
                time_scale,
            )
        )
        if leakage_only_resolved:
            model = "leakage_only"
            leakage_rate = one_rate
            seepage_rate = 0.0
            covariance = np.full((2, 2), np.nan)
            covariance[0, 0] = one_covariance[0, 0]
            leakage_error = one_error
            seepage_error = float("nan")
            fitted_ground, fitted_excited = _target_leakage_curves(
                time_array,
                initial_ground,
                initial_excited,
                leakage_rate,
                0.0,
            )
            success = bool(optimization.success)
            message = f"{optimization.message} Fallback: seepage fixed to zero."
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
            success = bool(optimization.success)
            message = f"{optimization.message} Fallback: no resolved target leakage."

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


def fit_cr_on_dephasing(
    model: _CrEchoDecayModel,
    n_values: ArrayLike,
    control_x: ArrayLike,
    target_z: ArrayLike,
    control_x_standard_errors: ArrayLike | None = None,
    target_z_standard_errors: ArrayLike | None = None,
) -> CrOnDephasingFit:
    """
    Jointly infer CR-on pure dephasing from control-X and target-Z curves.

    ``model`` supplies physical predictions for the complete C/D pulse
    blocks. Only the two CR-active pure-dephasing rates are optimized. An
    independent affine amplitude and offset are profiled for each observed
    curve to absorb SPAM, and their four degrees of freedom are included in
    the covariance estimate.
    """
    n_array = np.asarray(n_values)
    observed_control = np.asarray(control_x, dtype=np.float64)
    observed_target = np.asarray(target_z, dtype=np.float64)
    if n_array.ndim != 1 or observed_control.shape != n_array.shape:
        raise ValueError("n_values and control_x must be equal one-dimensional arrays.")
    if observed_target.shape != n_array.shape or n_array.size < 3:
        raise ValueError("target_z must match at least three n_values.")
    if not np.issubdtype(n_array.dtype, np.integer):
        raise ValueError("n_values must contain integers.")
    n_array = np.asarray(n_array, dtype=np.int64)
    if n_array[0] != 0 or np.any(np.diff(n_array) <= 0):
        raise ValueError("n_values must increase strictly from zero.")
    if not np.all(np.isfinite(observed_control)) or not np.all(
        np.isfinite(observed_target)
    ):
        raise ValueError("Echo-decay observations must contain only finite values.")
    control_errors = _resolve_standard_errors(
        control_x_standard_errors,
        observed_control.shape,
    )
    target_errors = _resolve_standard_errors(
        target_z_standard_errors,
        observed_target.shape,
    )
    rate_scale = model.cr_lobe_duration

    def evaluate(scaled_rates: NDArray[np.float64]):
        simulated_control, simulated_target = model.predict(
            n_array,
            float(scaled_rates[0] / rate_scale),
            float(scaled_rates[1] / rate_scale),
        )
        return (
            _profile_affine(simulated_control, observed_control, control_errors),
            _profile_affine(simulated_target, observed_target, target_errors),
        )

    def residual(scaled_rates: NDArray[np.float64]) -> NDArray[np.float64]:
        control_profile, target_profile = evaluate(scaled_rates)
        return np.concatenate((control_profile[3], target_profile[3]))

    optimization = least_squares(
        residual,
        x0=np.array([1e-3, 1e-3], dtype=np.float64),
        bounds=(0.0, np.inf),
        x_scale=1.0,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        max_nfev=80,
    )
    control_profile, target_profile = evaluate(optimization.x)
    covariance = (
        _estimate_covariance(
            np.asarray(optimization.jac, dtype=np.float64),
            float(optimization.cost),
            2 * n_array.size,
            _FORWARD_FIT_PARAMETER_COUNT,
            absolute_weights=control_errors is not None or target_errors is not None,
        )
        / rate_scale**2
    )
    parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    dense_n = np.arange(int(n_array[-1]) + 1, dtype=np.int64)
    dense_control, dense_target = model.predict(
        dense_n,
        float(optimization.x[0] / rate_scale),
        float(optimization.x[1] / rate_scale),
    )
    curve_control = control_profile[0] * dense_control + control_profile[1]
    curve_target = target_profile[0] * dense_target + target_profile[1]
    observed = np.concatenate((observed_control, observed_target))
    fitted = np.concatenate((control_profile[2], target_profile[2]))
    identifiable = np.linalg.matrix_rank(optimization.jac) == 2
    message = str(optimization.message)
    if not identifiable:
        message += " The two CR-on dephasing rates are not jointly identifiable."
    return CrOnDephasingFit(
        success=bool(optimization.success and identifiable),
        message=message,
        gamma_phi_control=float(optimization.x[0] / rate_scale),
        gamma_phi_rho_target=float(optimization.x[1] / rate_scale),
        gamma_phi_control_error=float(parameter_errors[0]),
        gamma_phi_rho_target_error=float(parameter_errors[1]),
        covariance=covariance,
        control_amplitude=control_profile[0],
        control_offset=control_profile[1],
        target_amplitude=target_profile[0],
        target_offset=target_profile[1],
        fitted_control_x=np.asarray(control_profile[2], dtype=np.float64),
        fitted_target_z=np.asarray(target_profile[2], dtype=np.float64),
        curve_n_values=dense_n,
        curve_control_x=np.asarray(curve_control, dtype=np.float64),
        curve_target_z=np.asarray(curve_target, dtype=np.float64),
        r_squared=_r_squared(observed, fitted),
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
        `leakage_model` records whether the full, outward-only, or no-leakage
        model was identifiable and used for the returned curves.
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
        outward_indices = (0, 1, 3)
        outward_optimization = optimize(outward_indices)
        if outward_optimization is None:
            return _failed_rate_fit(
                "The outward-only rate-model fit failed.",
                initial_ground,
                initial_excited,
                time_array.size,
            )
        outward_covariance = (
            _estimate_covariance(
                np.asarray(outward_optimization.jac, dtype=np.float64),
                float(outward_optimization.cost),
                residual_count,
                len(outward_indices),
                absolute_weights=errors_ground is not None
                or errors_excited is not None,
            )
            / time_scale**2
        )
        outward_rate = float(outward_optimization.x[2] / time_scale)
        outward_error = float(np.sqrt(max(outward_covariance[2, 2], 0.0)))
        outward_only_resolved = (
            outward_optimization.success
            and np.linalg.matrix_rank(outward_optimization.jac) == len(outward_indices)
            and outward_optimization.x[2] > 1e-8
            and np.isfinite(outward_error)
            and outward_error / outward_rate < threshold
        )
        if outward_only_resolved:
            leakage_model = "outward_only"
            active_indices = outward_indices
            fallback_message = "Unresolved control F-to-E seepage fixed to zero."
            optimization = outward_optimization
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


__all__ = [
    "CrOnDephasingFit",
    "ExponentialDecayFit",
    "TargetLeakageFit",
    "TargetLeakageModel",
    "TargetT1RhoFit",
    "ThreeLevelRateFit",
    "fit_cr_on_dephasing",
    "fit_exponential_decay",
    "fit_target_leakage",
    "fit_target_t1rho",
    "fit_three_level_rate_model",
]
