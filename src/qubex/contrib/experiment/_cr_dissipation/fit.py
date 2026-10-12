"""
Provide reusable ordinary-GLS primitives for CR dissipation fits.

The module builds whitened residual blocks, performs robust initialization
followed by ordinary GLS, profiles affine C/D nuisance parameters, selects
nested candidates by AICc, and reports local covariance estimates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares, lsq_linear
from scipy.stats import chi2

from .types import CandidateFit, FitGoodnessOfFit

_EPS = float(np.finfo(np.float64).eps)
GOODNESS_OF_FIT_REDUCED_CHI_SQUARED_LIMIT = 5.0
GOODNESS_OF_FIT_MAX_RESIDUAL_LIMIT = 8.0


# Residual construction and covariance whitening


@dataclass(frozen=True)
class ResidualBlock:
    """Store one retained residual block and its whitening matrix."""

    observed: NDArray[np.float64]
    whitener: NDArray[np.float64]
    label: str

    @property
    def rank(self) -> int:
        """Return the independent scalar residual dimension."""
        return int(self.whitener.shape[0])


def covariance_whitener(
    covariance: ArrayLike,
    *,
    rcond: float,
) -> tuple[NDArray[np.float64], int]:
    """Return a PSD pseudo-inverse square-root and effective rank."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be a finite square matrix.")
    scale = max(float(np.max(np.abs(cov))), _EPS)
    if not np.allclose(cov, cov.T, rtol=1e-7, atol=max(1e-15, scale * 1e-10)):
        raise ValueError("covariance must be symmetric within numerical tolerance.")
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), _EPS)
    cutoff = max(float(rcond), 0.0) * spectral_scale
    if float(np.min(eigenvalues)) < -max(cutoff, 1e-15 * spectral_scale):
        raise ValueError("covariance must be positive semidefinite.")
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError("covariance has no resolved positive mode.")
    whitener = (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])).T
    return np.asarray(whitener, dtype=np.float64), int(np.count_nonzero(keep))


def scalar_residual_blocks(
    values: ArrayLike,
    standard_errors: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
) -> tuple[ResidualBlock, ...]:
    """Build scalar residual blocks from finite positive standard errors."""
    observed = np.asarray(values, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if observed.ndim != 1 or errors.shape != observed.shape:
        raise ValueError("values and standard_errors must be matching 1D arrays.")
    point_labels = (
        tuple(str(index) for index in range(observed.size))
        if labels is None
        else tuple(labels)
    )
    if len(point_labels) != observed.size:
        raise ValueError("labels must match the number of scalar observations.")
    return tuple(
        ResidualBlock(
            observed=np.array([value], dtype=np.float64),
            whitener=np.array([[1.0 / error]], dtype=np.float64),
            label=label,
        )
        for value, error, label in zip(observed, errors, point_labels, strict=True)
        if np.isfinite(value) and np.isfinite(error) and error > 0.0
    )


def whiten_predictions(
    predictions: Sequence[ArrayLike],
    blocks: Sequence[ResidualBlock],
) -> NDArray[np.float64]:
    """Whiten predicted-minus-observed residual blocks."""
    if len(predictions) != len(blocks):
        raise ValueError("predictions must match retained residual blocks.")
    residuals = []
    for prediction, block in zip(predictions, blocks, strict=True):
        predicted = np.asarray(prediction, dtype=np.float64)
        if predicted.shape != block.observed.shape:
            raise ValueError("one prediction has an incompatible block shape.")
        residuals.append(block.whitener @ (predicted - block.observed))
    return np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float64)


def _run_least_squares(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    loss: str,
    max_nfev: int,
) -> OptimizeResult | None:
    """Run least squares and convert numerical failures to an unavailable fit."""
    try:
        return least_squares(
            residual,
            x0=initial,
            bounds=bounds,
            loss=loss,
            f_scale=1.0,
            max_nfev=max_nfev,
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _local_covariance(
    jacobian: NDArray[np.float64],
    *,
    reduced_chi_squared: float,
) -> NDArray[np.float64]:
    """Compute `(J_w.T J_w)+` once from a whitened-residual Jacobian."""
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        return np.full((jacobian.shape[-1], jacobian.shape[-1]), np.nan)
    n_parameters = jacobian.shape[1]
    if n_parameters == 0:
        return np.empty((0, 0), dtype=np.float64)
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((n_parameters, n_parameters), np.nan)
    if singular_values.size < n_parameters or not np.all(np.isfinite(singular_values)):
        return np.full((n_parameters, n_parameters), np.nan)
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    tolerance = max(
        1e-12 * largest,
        np.finfo(np.float64).eps * max(jacobian.shape) * largest,
    )
    if (
        largest <= 0.0
        or int(np.count_nonzero(singular_values > tolerance)) < n_parameters
    ):
        return np.full((n_parameters, n_parameters), np.nan)
    covariance = (vt.T / np.square(singular_values)) @ vt
    covariance = 0.5 * (covariance + covariance.T)
    return np.asarray(covariance * max(1.0, reduced_chi_squared), dtype=np.float64)


def assess_goodness_of_fit(
    whitened_residual: ArrayLike,
    *,
    n_parameters: int,
) -> FitGoodnessOfFit:
    """Assess gross model mismatch from absolute whitened residuals."""
    residual = np.asarray(whitened_residual, dtype=np.float64).reshape(-1)
    degrees_of_freedom = int(residual.size - n_parameters)
    if residual.size == 0 or not np.all(np.isfinite(residual)):
        return FitGoodnessOfFit(
            degrees_of_freedom,
            None,
            float("nan"),
            False,
            "No finite whitened residuals were available.",
        )
    chi_squared = float(residual @ residual)
    maximum = float(np.max(np.abs(residual)))
    if degrees_of_freedom <= 0:
        return FitGoodnessOfFit(
            degrees_of_freedom,
            None,
            maximum,
            False,
            "Absolute goodness-of-fit requires positive residual degrees of freedom.",
        )
    reduced = chi_squared / degrees_of_freedom
    p_value = float(chi2.sf(chi_squared, degrees_of_freedom))
    accepted = bool(
        reduced <= GOODNESS_OF_FIT_REDUCED_CHI_SQUARED_LIMIT
        and maximum <= GOODNESS_OF_FIT_MAX_RESIDUAL_LIMIT
    )
    reason = None
    if not accepted:
        reason = (
            "Whitened residuals exceed the conservative absolute fit limits "
            f"(reduced chi-squared={reduced:.3g}, max |r|={maximum:.3g})."
        )
    return FitGoodnessOfFit(
        degrees_of_freedom,
        p_value,
        maximum,
        accepted,
        reason,
    )


def _rate_multistart_initials(
    initial: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    names: Sequence[str],
) -> tuple[NDArray[np.float64], ...]:
    """Build deterministic low/high perturbations of free rate parameters."""
    rate_indices = tuple(
        index
        for index, name in enumerate(names)
        if "rate" in name or (name.startswith("control_") and "_to_" in name)
    )
    if not rate_indices:
        return ()

    def clipped(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(candidate, lower, upper)

    starts: list[NDArray[np.float64]] = []
    for factor in (0.1, 10.0):
        candidate = initial.copy()
        candidate[list(rate_indices)] *= factor
        starts.append(clipped(candidate))
    for index in rate_indices:
        candidate = initial.copy()
        candidate[index] *= 10.0
        starts.append(clipped(candidate))
    unique: list[NDArray[np.float64]] = []
    for candidate in starts:
        if not np.array_equal(candidate, initial) and not any(
            np.array_equal(candidate, existing) for existing in unique
        ):
            unique.append(candidate)
    return tuple(unique)


def fit_gls_candidate(
    *,
    name: str,
    parameter_names: Sequence[str],
    initial: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike],
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    prediction: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_observations: int,
    additional_initials: Sequence[ArrayLike] | None = None,
    preliminary_loss: str = "soft_l1",
    max_nfev: int = 3000,
) -> CandidateFit:
    """Fit one candidate with robust initialization and final ordinary GLS."""
    names = tuple(parameter_names)
    initial_array = np.asarray(initial, dtype=np.float64)
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    n_parameters = len(names)
    if (
        initial_array.shape != (n_parameters,)
        or lower.shape != initial_array.shape
        or upper.shape != initial_array.shape
    ):
        raise ValueError("candidate parameter arrays have incompatible shapes.")
    if n_parameters == 0:
        parameters = np.empty(0, dtype=np.float64)
        whitened = np.asarray(residual(parameters), dtype=np.float64)
        predicted = np.asarray(prediction(parameters), dtype=np.float64)
        success = bool(
            whitened.size == n_observations and np.all(np.isfinite(whitened))
        )
        chi_squared = float(whitened @ whitened) if success else float("nan")
        aicc = chi_squared if success and n_observations > 1 else None
        return CandidateFit(
            name=name,
            success=success,
            parameter_names=names,
            parameters=parameters,
            covariance=np.empty((0, 0), dtype=np.float64),
            standard_errors=np.empty(0, dtype=np.float64),
            jacobian=np.empty((n_observations, 0), dtype=np.float64),
            prediction=predicted,
            chi_squared=chi_squared,
            reduced_chi_squared=(chi_squared / max(1, n_observations))
            if success
            else float("nan"),
            aicc=aicc,
            n_observations=n_observations,
            n_parameters=0,
            message="Fixed-parameter candidate evaluated."
            if success
            else "Fixed-parameter candidate evaluation failed.",
            goodness_of_fit=(
                assess_goodness_of_fit(whitened, n_parameters=0) if success else None
            ),
        )

    if n_observations <= n_parameters + 1:
        return _failed_candidate(
            name,
            names,
            n_observations,
            "AICc is undefined because N_obs <= k + 1.",
        )
    extra = (
        _rate_multistart_initials(initial_array, lower, upper, names)
        if additional_initials is None
        else tuple(np.asarray(value, dtype=np.float64) for value in additional_initials)
    )
    starts = (initial_array, *extra)
    if any(start.shape != initial_array.shape for start in starts):
        raise ValueError("all candidate initializations must have matching shapes.")
    if any(np.any(start < lower) or np.any(start > upper) for start in starts):
        raise ValueError("candidate initializations must lie within bounds.")
    solutions: list[tuple[float, OptimizeResult, NDArray[np.float64]]] = []
    for start in starts:
        robust = _run_least_squares(
            residual,
            start,
            (lower, upper),
            loss=preliminary_loss,
            max_nfev=max_nfev,
        )
        final_initial = (
            np.asarray(robust.x, dtype=np.float64)
            if robust is not None and np.all(np.isfinite(robust.x))
            else start
        )
        final_attempt = _run_least_squares(
            residual,
            final_initial,
            (lower, upper),
            loss="linear",
            max_nfev=max_nfev,
        )
        if (
            final_attempt is None
            or not final_attempt.success
            or not np.all(np.isfinite(final_attempt.x))
        ):
            continue
        attempt_residual = np.asarray(
            residual(np.asarray(final_attempt.x, dtype=np.float64)), dtype=np.float64
        )
        if attempt_residual.shape != (n_observations,) or not np.all(
            np.isfinite(attempt_residual)
        ):
            continue
        solutions.append(
            (
                float(attempt_residual @ attempt_residual),
                final_attempt,
                attempt_residual,
            )
        )
    if not solutions:
        return _failed_candidate(
            name,
            names,
            n_observations,
            "Final ordinary GLS optimization failed.",
        )
    _, final, whitened = min(solutions, key=lambda item: item[0])
    parameters = np.asarray(final.x, dtype=np.float64)
    chi_squared = float(whitened @ whitened)
    reduced = chi_squared / max(1, n_observations - n_parameters)
    jacobian = np.asarray(final.jac, dtype=np.float64)
    covariance = _local_covariance(jacobian, reduced_chi_squared=reduced)
    if not np.all(np.isfinite(covariance)):
        return _failed_candidate(
            name,
            names,
            n_observations,
            "Final ordinary GLS parameters are not locally identifiable.",
        )
    standard_errors = (
        np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if covariance.shape == (n_parameters, n_parameters)
        else np.full(n_parameters, np.nan)
    )
    aicc = (
        chi_squared
        + 2.0 * n_parameters
        + 2.0 * n_parameters * (n_parameters + 1) / (n_observations - n_parameters - 1)
    )
    return CandidateFit(
        name=name,
        success=True,
        parameter_names=names,
        parameters=parameters,
        covariance=covariance,
        standard_errors=np.asarray(standard_errors, dtype=np.float64),
        jacobian=jacobian,
        prediction=np.asarray(prediction(parameters), dtype=np.float64),
        chi_squared=chi_squared,
        reduced_chi_squared=float(reduced),
        aicc=float(aicc),
        n_observations=n_observations,
        n_parameters=n_parameters,
        message=str(final.message),
        goodness_of_fit=assess_goodness_of_fit(whitened, n_parameters=n_parameters),
        n_initializations=len(starts),
        n_successful_initializations=len(solutions),
    )


def fit_profiled_affine_candidate(
    *,
    name: str,
    simulator: Callable[[float], NDArray[np.float64]],
    values: ArrayLike,
    standard_errors: ArrayLike,
    fixed_rate: float | None = None,
    initial_rate: float | None = None,
    rate_bounds: tuple[float, float] | None = None,
    max_nfev: int = 300,
) -> CandidateFit:
    """Fit one nonlinear rate after profiling affine amplitude and offset."""
    observed = np.asarray(values, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if observed.ndim != 1 or errors.shape != observed.shape:
        raise ValueError("values and standard_errors must be matching 1D arrays.")
    retained = np.flatnonzero(
        np.isfinite(observed) & np.isfinite(errors) & (errors > 0.0)
    )
    n_observations = int(retained.size)
    free_rate = fixed_rate is None
    parameter_names = (
        ("pure_dephasing_rate_per_ns", "spam_amplitude", "spam_offset")
        if free_rate
        else ("spam_amplitude", "spam_offset")
    )
    n_parameters = len(parameter_names)
    if n_observations <= n_parameters + 1:
        return _failed_candidate(
            name,
            parameter_names,
            n_observations,
            "AICc is undefined because N_obs <= k + 1.",
        )

    def evaluate_simulator(rate: float) -> NDArray[np.float64]:
        """Return one validated simulator curve."""
        simulated = np.asarray(simulator(float(rate)), dtype=np.float64)
        if simulated.shape != observed.shape or not np.all(
            np.isfinite(simulated[retained])
        ):
            raise ValueError("The profiled simulator returned an invalid curve.")
        return simulated

    def profile(
        rate: float,
    ) -> tuple[
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        simulated = evaluate_simulator(rate)
        design = np.column_stack(
            (simulated[retained] / errors[retained], 1.0 / errors[retained])
        )
        target = observed[retained] / errors[retained]
        linear = lsq_linear(
            design,
            target,
            bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        )
        if not linear.success or not np.all(np.isfinite(linear.x)):
            raise ValueError("The profiled affine GLS solve failed.")
        amplitude, offset = (float(linear.x[0]), float(linear.x[1]))
        prediction = amplitude * simulated + offset
        residual = (prediction[retained] - observed[retained]) / errors[retained]
        return amplitude, offset, prediction, residual, simulated

    if free_rate:
        if initial_rate is None or rate_bounds is None:
            raise ValueError("A free profiled rate requires initial_rate and bounds.")
        lower, upper = (float(rate_bounds[0]), float(rate_bounds[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("rate_bounds must be finite and strictly increasing.")
        initial = float(np.clip(initial_rate, lower, upper))

        def residual(rate_array: NDArray[np.float64]) -> NDArray[np.float64]:
            return profile(float(rate_array[0]))[3]

        lower_array = np.array([lower])
        upper_array = np.array([upper])
        start_values = tuple(
            dict.fromkeys(
                (
                    initial,
                    lower + 0.1 * (upper - lower),
                    lower + 0.5 * (upper - lower),
                    lower + 0.9 * (upper - lower),
                )
            )
        )
        solutions: list[tuple[float, OptimizeResult]] = []
        for start in start_values:
            robust = _run_least_squares(
                residual,
                np.array([start]),
                (lower_array, upper_array),
                loss="soft_l1",
                max_nfev=max_nfev,
            )
            final_initial = (
                np.asarray(robust.x, dtype=np.float64)
                if robust is not None and np.all(np.isfinite(robust.x))
                else np.array([start])
            )
            attempt = _run_least_squares(
                residual,
                final_initial,
                (lower_array, upper_array),
                loss="linear",
                max_nfev=max_nfev,
            )
            if (
                attempt is None
                or not attempt.success
                or not np.all(np.isfinite(attempt.x))
            ):
                continue
            attempt_residual = residual(np.asarray(attempt.x, dtype=np.float64))
            if np.all(np.isfinite(attempt_residual)):
                solutions.append((float(attempt_residual @ attempt_residual), attempt))
        if not solutions:
            return _failed_candidate(
                name,
                parameter_names,
                n_observations,
                "Profiled one-dimensional GLS optimization failed.",
            )
        _, final = min(solutions, key=lambda item: item[0])
        rate = float(final.x[0])
    else:
        rate = float(fixed_rate)
        lower = rate
        upper = rate

    try:
        amplitude, offset, prediction, whitened, simulated = profile(rate)
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return _failed_candidate(
            name,
            parameter_names,
            n_observations,
            "Profiled affine GLS evaluation failed.",
        )
    chi_squared = float(whitened @ whitened)
    reduced = chi_squared / max(1, n_observations - n_parameters)
    if free_rate:
        step = max(1e-10, 1e-5 * max(abs(rate), upper - lower))
        left = max(lower, rate - step)
        right = min(upper, rate + step)
        if right <= left:
            return _failed_candidate(
                name,
                parameter_names,
                n_observations,
                "Profiled rate derivative could not be evaluated.",
            )
        try:
            left_curve = evaluate_simulator(left)
            right_curve = evaluate_simulator(right)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return _failed_candidate(
                name,
                parameter_names,
                n_observations,
                "Profiled rate derivative could not be evaluated.",
            )
        derivative = (right_curve - left_curve) / (right - left)
        jacobian = np.column_stack(
            (
                amplitude * derivative[retained] / errors[retained],
                simulated[retained] / errors[retained],
                1.0 / errors[retained],
            )
        )
        parameters = np.array([rate, amplitude, offset], dtype=np.float64)
    else:
        jacobian = np.column_stack(
            (simulated[retained] / errors[retained], 1.0 / errors[retained])
        )
        parameters = np.array([amplitude, offset], dtype=np.float64)
    covariance = _local_covariance(jacobian, reduced_chi_squared=reduced)
    if not np.all(np.isfinite(covariance)):
        return _failed_candidate(
            name,
            parameter_names,
            n_observations,
            "Profiled GLS parameters are not locally identifiable.",
        )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    aicc = (
        chi_squared
        + 2.0 * n_parameters
        + 2.0 * n_parameters * (n_parameters + 1) / (n_observations - n_parameters - 1)
    )
    return CandidateFit(
        name=name,
        success=True,
        parameter_names=parameter_names,
        parameters=parameters,
        covariance=covariance,
        standard_errors=np.asarray(standard_errors, dtype=np.float64),
        jacobian=np.asarray(jacobian, dtype=np.float64),
        prediction=prediction,
        chi_squared=chi_squared,
        reduced_chi_squared=float(reduced),
        aicc=float(aicc),
        n_observations=n_observations,
        n_parameters=n_parameters,
        message="Affine SPAM parameters were profiled in a one-dimensional rate fit.",
        goodness_of_fit=assess_goodness_of_fit(whitened, n_parameters=n_parameters),
        n_initializations=len(start_values) if free_rate else 1,
        n_successful_initializations=len(solutions) if free_rate else 1,
    )


def _failed_candidate(
    name: str,
    parameter_names: tuple[str, ...],
    n_observations: int,
    message: str,
) -> CandidateFit:
    """Build a consistently shaped failed candidate result."""
    n_parameters = len(parameter_names)
    return CandidateFit(
        name=name,
        success=False,
        parameter_names=parameter_names,
        parameters=np.full(n_parameters, np.nan),
        covariance=np.full((n_parameters, n_parameters), np.nan),
        standard_errors=np.full(n_parameters, np.nan),
        jacobian=np.full((n_observations, n_parameters), np.nan),
        prediction=np.empty(0, dtype=np.float64),
        chi_squared=float("nan"),
        reduced_chi_squared=float("nan"),
        aicc=None,
        n_observations=n_observations,
        n_parameters=n_parameters,
        message=message,
    )


def select_aicc_candidate(
    candidates: Mapping[str, CandidateFit],
    *,
    delta_threshold: float = 6.0,
) -> CandidateFit | None:
    """Select the simplest candidate within the AICc evidence threshold."""
    eligible = [
        candidate
        for candidate in candidates.values()
        if candidate.success
        and candidate.aicc is not None
        and np.isfinite(candidate.aicc)
    ]
    if not eligible:
        return None
    minimum = min(
        float(candidate.aicc) for candidate in eligible if candidate.aicc is not None
    )
    competitive = [
        candidate
        for candidate in eligible
        if candidate.aicc is not None and candidate.aicc <= minimum + delta_threshold
    ]
    return min(
        competitive,
        key=lambda candidate: (
            candidate.n_parameters,
            float(candidate.aicc) if candidate.aicc is not None else float("inf"),
            candidate.name,
        ),
    )
