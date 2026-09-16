"""Ordinary-GLS fitting utilities for CR dissipation analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares

from ._cr_dissipation_types import CandidateFit

_EPS = float(np.finfo(np.float64).eps)


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


def fit_gls_candidate(
    *,
    name: str,
    parameter_names: Sequence[str],
    initial: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike],
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    prediction: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_observations: int,
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
        )

    if n_observations <= n_parameters + 1:
        return _failed_candidate(
            name,
            names,
            n_observations,
            "AICc is undefined because N_obs <= k + 1.",
        )
    robust = _run_least_squares(
        residual,
        initial_array,
        (lower, upper),
        loss=preliminary_loss,
        max_nfev=max_nfev,
    )
    final_initial = (
        np.asarray(robust.x, dtype=np.float64)
        if robust is not None and np.all(np.isfinite(robust.x))
        else initial_array
    )
    final = _run_least_squares(
        residual,
        final_initial,
        (lower, upper),
        loss="linear",
        max_nfev=max_nfev,
    )
    if final is None or not final.success or not np.all(np.isfinite(final.x)):
        return _failed_candidate(
            name,
            names,
            n_observations,
            "Final ordinary GLS optimization failed.",
        )
    parameters = np.asarray(final.x, dtype=np.float64)
    whitened = np.asarray(residual(parameters), dtype=np.float64)
    if whitened.shape != (n_observations,) or not np.all(np.isfinite(whitened)):
        return _failed_candidate(
            name,
            names,
            n_observations,
            "Final ordinary GLS residual is invalid.",
        )
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
    )


def _failed_candidate(
    name: str,
    parameter_names: tuple[str, ...],
    n_observations: int,
    message: str,
) -> CandidateFit:
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


def covariance_standard_errors(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return finite covariance diagonal errors and preserve unavailable entries."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    diagonal = np.diag(covariance)
    return np.where(
        np.isfinite(diagonal) & (diagonal >= 0.0),
        np.sqrt(np.maximum(diagonal, 0.0)),
        np.nan,
    )
