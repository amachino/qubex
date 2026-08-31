"""
Offline, phenomenological joint-chevron fit; never a gate-fidelity estimator.

Shared carrier center, cyclic oscillation rate, ramp-time offset and local
detuning slope; separate offsets, contrasts and decay times for the two
prepared-state branches. Strong-drive slope is fitted, not fixed by photon
count. Frequencies/rates are MHz, durations are ns. Model mismatch, SPAM and
natural drift are not removed by this fit. No hardware or file I/O.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

PARAMETER_NAMES = (
    "center_offset_mhz",
    "rate_mhz",
    "time_offset_ns",
    "detuning_slope",
    "offset_00",
    "offset_11",
    "visibility_00",
    "visibility_11",
    "decay_00_ns",
    "decay_11_ns",
)


def chevron_model(
    parameters: ArrayLike, frequency_offsets_mhz: ArrayLike, durations_ns: ArrayLike
) -> NDArray[np.float64]:
    """Return shape (2, frequency, time) with positive transfer contrast."""
    p = np.asarray(parameters, dtype=float)
    f = np.asarray(frequency_offsets_mhz, dtype=float)[:, None]
    t = np.asarray(durations_ns, dtype=float)[None, :]
    detuning = p[3] * (f - p[0])
    rate = np.sqrt(p[1] ** 2 + detuning**2)
    elapsed = np.maximum(t - p[2], 0.0)
    oscillation = (p[1] / rate) ** 2 * np.sin(np.pi * rate * elapsed / 1000) ** 2
    return np.stack(
        [
            p[4 + d] + (1 - p[4 + d]) * p[6 + d] * np.exp(-t / p[8 + d]) * oscillation
            for d in range(2)
        ]
    )


def fit_bidirectional_chevron(
    frequencies_ghz: ArrayLike,
    durations_ns: ArrayLike,
    success_counts: ArrayLike,
    shots: int,
    *,
    ramp_ns: float = 16.0,
) -> dict[str, Any]:
    """
    Fit raw binary success counts; return diagnostic parameters and arrays.

    Parameters
    ----------
    frequencies_ghz : array_like
        Increasing carrier frequencies in GHz, with at least seven values.
    durations_ns : array_like
        Increasing complete-pulse durations in ns, with at least twelve values.
    success_counts : array_like
        Unmitigated transfer counts, shape (2, frequency, duration).
    shots : int
        Positive shots per sample and direction.
    ramp_ns : float, default 16
        Positive duration of each ramp in ns.

    Returns
    -------
    dict
        Resonance candidate, effective parameters, uncertainties and residuals.

    Notes
    -----
    Missing samples are NaN or negative. Local-Jacobian standard errors are
    approximations conditional on the model and fixed classifier, not full
    calibration confidence intervals. Inspect both surfaces and residuals;
    independent repeat measurements and native-grid tests remain necessary.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=float)
    durations = np.asarray(durations_ns, dtype=float)
    counts = np.asarray(success_counts, dtype=float)
    if frequencies.ndim != 1 or durations.ndim != 1:
        raise ValueError("Frequency and time coordinates must be one-dimensional")
    if len(frequencies) < 7 or len(durations) < 12:
        raise ValueError(
            "Need at least 7 frequencies and 12 durations for this joint fit"
        )
    if not np.all(np.isfinite(frequencies)) or not np.all(np.diff(frequencies) > 0):
        raise ValueError("Frequency coordinates must be finite and strictly increasing")
    if (
        not np.all(np.isfinite(durations))
        or not np.all(np.diff(durations) > 0)
        or durations[0] < 0
    ):
        raise ValueError(
            "Duration coordinates must be finite, nonnegative and increasing"
        )
    if counts.shape != (2, len(frequencies), len(durations)):
        raise ValueError("Success counts must have shape (2, frequency, duration)")
    if not np.isfinite(shots) or shots <= 0 or int(shots) != shots:
        raise ValueError("shots must be a positive integer")
    if not np.isfinite(ramp_ns) or ramp_ns <= 0:
        raise ValueError("ramp_ns must be positive")
    finite = np.isfinite(counts) & (counts >= 0)
    if np.any(counts[finite] > shots) or np.any(
        counts[finite] != np.rint(counts[finite])
    ):
        raise ValueError("Success counts must be integer values in [0, shots]")
    if np.any(np.sum(finite, axis=(1, 2)) < 0.8 * len(frequencies) * len(durations)):
        raise ValueError("This fit requires at least 80% coverage in each direction")
    y = counts / shots
    y[~finite] = np.nan
    contrast = np.nanpercentile(y, 95, axis=(1, 2)) - np.nanpercentile(
        y, 5, axis=(1, 2)
    )
    if np.any(contrast < 0.05):
        raise ValueError("Insufficient visible contrast for this effective-chevron fit")

    origin = float(np.mean(frequencies))
    f = (frequencies - origin) * 1000
    # Jeffreys-binomial variance supplies finite weights at zero/full counts.
    weighted_counts = np.where(finite, counts, 0.0)
    sigma = np.sqrt(
        (weighted_counts + 0.5)
        * (shots - weighted_counts + 0.5)
        / ((shots + 1) ** 2 * (shots + 2))
    )

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((chevron_model(p, f, durations) - y) / sigma)[finite]

    lower = np.array([f[0], 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 200.0, 200.0])
    upper = np.array(
        [f[-1], 8.0, 2 * ramp_ns, 4.0, 0.8, 0.8, 1.0, 1.0, 100000.0, 100000.0]
    )
    scales = np.array(
        [max(np.ptp(f) / 4, 0.1), 2.0, ramp_ns, 1.0, 0.1, 0.1, 0.7, 0.7, 5000.0, 5000.0]
    )
    offset = np.clip(np.nanpercentile(y, 5, axis=(1, 2)), 0.005, 0.5)
    visibility = np.clip(contrast / (1 - offset), 0.1, 0.98)
    solutions = []
    for center in (0.0, f[0] / 2, f[-1] / 2):
        for rate in (1.0, 2.0, 4.0):
            seed = [center, rate, ramp_ns, 1.5, *offset, *visibility, 5000.0, 5000.0]
            fit = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                x_scale=cast(
                    Any, scales
                ),  # SciPy accepts an array; its stub is scalar-only.
                loss="soft_l1",
                max_nfev=1500,
            )
            if fit.success and np.all(np.isfinite(fit.x)):
                solutions.append(fit)
    if not solutions:
        raise ValueError("No multistart fit converged")
    best = min(solutions, key=lambda fit: fit.cost)
    p = best.x
    predicted = chevron_model(p, f, durations)
    dof = int(finite.sum() - len(p))
    reduced_chi2 = float(np.sum(residual(p) ** 2) / dof)
    rank = int(np.linalg.matrix_rank(best.jac))
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * max(1.0, reduced_chi2)
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0))
    warnings = []
    if rank < len(p):
        warnings.append("Rank-deficient local Jacobian; parameters are not identified")
    if reduced_chi2 > 3:
        warnings.append(
            "Large residual versus shot noise; effective model needs review"
        )
    warnings.extend(
        f"Shared parameter near fit bound: {PARAMETER_NAMES[i]}"
        for i in range(4)
        if min(p[i] - lower[i], upper[i] - p[i]) < 0.01 * (upper[i] - lower[i])
    )
    if stderr[0] > np.ptp(f) / 4:
        warnings.append("Large local uncertainty in the resonance center")
    near = [s for s in solutions if s.cost <= best.cost + max(1.0, 0.01 * best.cost)]
    if np.ptp([s.x[0] for s in near]) > max(0.1, 2 * stderr[0]):
        warnings.append("Comparable minima disagree on the resonance center")
    if (durations[-1] - p[2]) * p[1] / 1000 < 2:
        warnings.append("Time span covers fewer than two fitted oscillation periods")
    full_time = float(p[2] + 500 / p[1])
    half_time = float(p[2] + 250 / p[1])
    grid_seed = int(2 * round(full_time / 2))
    return {
        "parameters": dict(zip(PARAMETER_NAMES, map(float, p), strict=True)),
        "local_standard_errors": dict(
            zip(PARAMETER_NAMES, map(float, stderr), strict=True)
        ),
        "frequency_origin_ghz": origin,
        "resonance_frequency_ghz": origin + float(p[0]) / 1000,
        "resonance_local_standard_error_mhz": float(stderr[0]),
        "full_exchange_phase_time_seed_ns": full_time,
        "half_exchange_phase_time_seed_ns": half_time,
        "full_exchange_native_neighbors_ns": [grid_seed - 2, grid_seed, grid_seed + 2],
        "reduced_chi2": reduced_chi2,
        "jacobian_rank": rank,
        "warnings": warnings,
        "diagnostic_only": True,
        "uncertainty_scope": "local Jacobian, fixed classifier and model; excludes drift and SPAM uncertainty",
        "time_scope": "phase-angle seeds, not measured optimal durations or gate fidelities",
        "observed": y,
        "predicted": predicted,
        "residual": y - predicted,
    }
