"""Damped bidirectional duration fit with a fixed-ramp phase offset; offline only."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


def damped_transfer(
    parameters: ArrayLike, flat_time_ns: ArrayLike
) -> NDArray[np.float64]:
    """Return two population curves with common oscillation rate and ramp phase."""
    p = np.asarray(parameters)
    time_us = np.asarray(flat_time_ns) / 1000
    angle = 2 * np.pi * p[0] * time_us + p[1]
    return np.stack(
        [
            p[2 + d]
            + (1 - p[2 + d])
            * p[4 + d]
            * (1 - np.exp(-p[6 + d] * time_us) * np.cos(angle))
            / 2
            for d in range(2)
        ]
    )


def fit_damped_duration(
    durations_ns: ArrayLike,
    success_counts: ArrayLike,
    shots: int,
    *,
    ramp_ns: float,
    rate_range_mhz: tuple[float, float] = (0.5, 5.0),
    grid_ns: float = 2.0,
) -> dict[str, Any]:
    """
    Infer full and half-exchange phase times from a damped dwell oscillation.

    Parameters
    ----------
    durations_ns : array_like
        Increasing complete-pulse durations in ns, including both ramps.
    success_counts : array_like
        Bidirectional transfer counts, shape (2, duration). NaNs mark missing points.
    shots : int
        Positive shots per point and direction.
    ramp_ns : float
        Positive per-side ramp duration in ns.
    rate_range_mhz : tuple of float, default (0.5, 5.0)
        Lower and upper cyclic oscillation-rate bounds in MHz.
    grid_ns : float, default 2
        Positive native duration grid spacing in ns.

    Returns
    -------
    dict
        First full/root phase-angle candidates, rounded durations, neighbors,
        local uncertainties and residuals. These are not damped maxima or fidelity.
    """
    times = np.asarray(durations_ns, dtype=float)
    counts = np.asarray(success_counts, dtype=float)
    flat = times - 2 * ramp_ns
    if times.ndim != 1 or counts.shape != (2, len(times)):
        raise ValueError("Need durations and counts with shape (2, duration)")
    if (
        not np.all(np.isfinite(times))
        or shots <= 0
        or ramp_ns <= 0
        or grid_ns <= 0
        or np.any(flat < 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Need increasing completed-pulse durations >= 2*ramp and positive shots/grid"
        )
    finite = np.isfinite(counts)
    if (
        np.any(finite.sum(axis=1) < 20)
        or np.any(counts[finite] < 0)
        or np.any(counts[finite] > shots)
    ):
        raise ValueError(
            "Need at least 20 valid points per direction and counts in [0, shots]"
        )
    lo, hi = rate_range_mhz
    if not 0 < lo < hi or hi * np.max(np.diff(times)) / 1000 >= 0.5:
        raise ValueError(
            "Rate search must be positive and below the sampling Nyquist limit"
        )
    observed = counts / shots
    sigma = np.sqrt(np.maximum(observed * (1 - observed), 0.25 / shots) / shots)
    lower = [lo, -np.pi, 0, 0, 0, 0, 0, 0]
    upper = [hi, np.pi, 1, 1, 1, 1, 10, 10]
    # Linear cosine/sine fits cheaply locate frequency/phase initial guesses.
    seeds = []
    for rate in np.linspace(lo, hi, 81):
        angle = 2 * np.pi * rate * flat / 1000
        matrix = np.column_stack([np.ones(len(flat)), np.cos(angle), np.sin(angle)])
        coefs = [
            np.linalg.lstsq(matrix[finite[d]], observed[d, finite[d]], rcond=None)[0]
            for d in range(2)
        ]
        rss = sum(
            np.sum((matrix[finite[d]] @ coefs[d] - observed[d, finite[d]]) ** 2)
            for d in range(2)
        )
        mean_coef = np.mean(coefs, axis=0)
        phase = np.arctan2(mean_coef[2], -mean_coef[1])
        seeds.append((rss, rate, phase))

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((damped_transfer(p, flat) - observed) / sigma)[finite]

    fits = []
    for _, rate, phase in sorted(seeds)[:6]:
        floor = np.clip(np.nanmin(observed, axis=1), 0.001, 0.4)
        visibility = np.clip(
            np.nanmax(observed, axis=1) - np.nanmin(observed, axis=1), 0.05, 0.95
        )
        fit = least_squares(
            residual,
            [rate, phase, *floor, *visibility, 0.1, 0.1],
            bounds=(lower, upper),
            max_nfev=4000,
        )
        if fit.success:
            fits.append(fit)
    if not fits:
        raise ValueError("Damped oscillation fit did not converge")
    fit = min(fits, key=lambda f: np.sum(f.fun**2))
    p = fit.x
    if p[0] * np.ptp(flat) / 1000 < 1:
        raise ValueError(
            "Less than one fitted oscillation is covered; extend the duration scan"
        )
    dof = finite.sum() - len(p)
    chi2 = float(np.sum(fit.fun**2) / dof)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * max(1, chi2)
    warnings = []
    if chi2 > 2:
        warnings.append("Residuals exceed shot noise; inspect the effective-model fit")
    if np.linalg.matrix_rank(fit.jac) < len(p):
        warnings.append(
            "Fit Jacobian is rank deficient; local uncertainties are unreliable"
        )
    if min(p[0] - lo, hi - p[0]) < 0.01 * (hi - lo):
        warnings.append("Frequency is near a search boundary")
    if min(p[4:6]) < 0.1:
        warnings.append("Low fitted contrast; phase-angle candidates are unreliable")

    def candidate(target_angle: float) -> dict[str, Any]:
        dwell = (target_angle - p[1]) / (2 * np.pi * p[0]) * 1000
        duration = 2 * ramp_ns + dwell
        if dwell < 0 or not times[0] <= duration <= times[-1]:
            return {
                "available": False,
                "reason": "first target phase is outside the measured nonnegative dwell range",
            }
        gradient = np.zeros(len(p))
        gradient[0] = -dwell / p[0]
        gradient[1] = -1000 / (2 * np.pi * p[0])
        rounded = grid_ns * np.floor(duration / grid_ns + 0.5)
        neighbors = [
            float(t)
            for t in (rounded - grid_ns, rounded, rounded + grid_ns)
            if times[0] <= t <= times[-1]
        ]
        return dict(
            available=True,
            flat_duration_ns=float(dwell),
            duration_ns=float(duration),
            grid_duration_ns=float(rounded),
            neighbors_ns=neighbors,
            local_standard_error_ns=float(
                np.sqrt(max(0, gradient @ covariance @ gradient))
            ),
        )

    full, half = candidate(np.pi), candidate(np.pi / 2)
    if not full["available"] or not half["available"]:
        warnings.append(
            "A first full/root phase target is unavailable; do not infer it from a later cycle"
        )
    names = [
        "rate_mhz",
        "ramp_phase_rad",
        "offset_00",
        "offset_11",
        "visibility_00",
        "visibility_11",
        "decay_rate_00_per_us",
        "decay_rate_11_per_us",
    ]
    return dict(
        parameters=dict(zip(names, map(float, p), strict=True)),
        decay_times_ns=[float(1000 / g) if g > 1e-8 else None for g in p[6:8]],
        local_standard_errors=dict(
            zip(
                names,
                map(float, np.sqrt(np.maximum(0, np.diag(covariance)))),
                strict=True,
            )
        ),
        bswap=full,
        sqrt_bswap=half,
        reduced_chi2=chi2,
        warnings=warnings,
        ramp_ns=float(ramp_ns),
        grid_ns=float(grid_ns),
        phase_scope="principal ramp phase modulo 2*pi; first-cycle phase-angle candidates",
        uncertainty_scope="local Jacobian approximation; excludes model error, SPAM and drift",
        predicted=damped_transfer(p, flat),
        residual=observed - damped_transfer(p, flat),
    )
