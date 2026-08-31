"""
Offline, phenomenological joint-chevron fit; never a gate-fidelity estimator.

Shared carrier center, cyclic oscillation rate, ramp-time offset and local
detuning slope; separate offsets, contrasts and decay times for the two
prepared-state branches. Strong-drive slope is fitted, not fixed by photon
count. Frequencies/rates are MHz, durations are ns. Model mismatch, SPAM and
natural drift are not removed by this fit. No hardware or file I/O.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .duration_fit import damped_transfer

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

_DAMPED_PARAMETER_NAMES = (
    "rate_mhz",
    "ramp_phase_rad",
    "offset_00",
    "offset_11",
    "visibility_00",
    "visibility_11",
    "decay_rate_00_per_us",
    "decay_rate_11_per_us",
)
_TOLERATED_OPERATIONAL_WARNING = (
    "Residuals exceed shot noise; inspect the effective-model fit"
)


def select_local_operational_candidate(
    frequencies_ghz: ArrayLike,
    row_fits: Sequence[Mapping[str, Any]],
    *,
    seed_frequency_ghz: float,
    rate_range_mhz: tuple[float, float] = (0.3, 3.0),
    max_row_reduced_chi2: float = 3.0,
    minimum_visibility: float = 0.6,
    maximum_span_mhz: float = 1.0,
) -> dict[str, Any]:
    """
    Select one measured local row for an independent operational holdout.

    Parameters
    ----------
    frequencies_ghz : array_like
        Exactly five increasing measured carrier frequencies in cyclic GHz.
        Their total span must not exceed `maximum_span_mhz`.
    row_fits : sequence of Mapping
        One `fit_damped_duration` result per frequency, using both directions,
        complete pulse durations, and the same rate/ramp/grid convention.
    seed_frequency_ghz : float
        Finite provenance seed used only to break exact score ties.
    rate_range_mhz : tuple of float, optional
        Strict interior range for every shared fitted cyclic rate, default
        (0.3, 3.0) MHz.
    max_row_reduced_chi2 : float, optional
        Explicit operational candidate-generation limit, default 3. A row's
        residual-above-shot-noise warning is still retained when chi-square is
        between the duration helper's warning threshold and this limit.
    minimum_visibility : float, optional
        Minimum fitted visibility in each direction and every row, default 0.6.
    maximum_span_mhz : float, optional
        Largest permitted five-row frequency span, default 1 MHz.

    Returns
    -------
    dict[str, Any]
        All row diagnostics and, only when every row passes, an
        `operational_response_candidate` chosen by minimum predicted
        directional transfer, then mean transfer, seed distance and lower
        frequency. This generates a holdout candidate; it never qualifies or
        adopts a gate.

    Raises
    ------
    ValueError
        Malformed frequencies, thresholds, fit schema, native-grid duration,
        or nonfinite numerical inputs.
    TypeError
        A row availability or warning field has the wrong type.

    Notes
    -----
    This function deliberately fits no cross-frequency resonance, gap or
    Hamiltonian curve. Each prediction is the supplied row's damped model at
    its own first-full 2 ns candidate. Failed rows are never dropped. The
    returned candidate requires fresh, independently acquired frequency and
    duration holdout checks before calibration or gate adoption.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=float)
    if (
        frequencies.shape != (5,)
        or not np.isfinite(frequencies).all()
        or np.any(np.diff(frequencies) <= 0)
        or np.any(frequencies <= 0)
        or np.any(frequencies > 20)
    ):
        raise ValueError("Need exactly five increasing finite frequency values in GHz")
    if not np.isfinite(seed_frequency_ghz) or not 0 < seed_frequency_ghz <= 20:
        raise ValueError("seed_frequency_ghz must be finite and in GHz")
    lo, hi = map(float, rate_range_mhz)
    limit = float(max_row_reduced_chi2)
    visibility_limit = float(minimum_visibility)
    span_limit = float(maximum_span_mhz)
    if (
        not np.isfinite([lo, hi, limit, visibility_limit, span_limit]).all()
        or not 0 < lo < hi
        or limit <= 0
        or not 0 < visibility_limit <= 1
        or span_limit <= 0
    ):
        raise ValueError("Invalid operational row-fit bounds or limits")
    if np.ptp(frequencies) * 1000 > span_limit + 1e-12:
        raise ValueError("The five frequency values exceed the local span in MHz")
    if len(row_fits) != 5:
        raise ValueError("Need exactly five row fits for the five frequency values")

    diagnostics = []
    reasons = []
    retained_warnings: list[str] = []
    common_ramp = None
    for index, (frequency, fit) in enumerate(zip(frequencies, row_fits, strict=True)):
        try:
            parameters = fit["parameters"]
            errors = fit["local_standard_errors"]
            full = fit["bswap"]
            ramp = float(fit["ramp_ns"])
            reduced_chi2 = float(fit["reduced_chi2"])
            rate = float(parameters["rate_mhz"])
            rate_error = float(errors["rate_mhz"])
            visibility = [
                float(parameters["visibility_00"]),
                float(parameters["visibility_11"]),
            ]
            available = full["available"]
            duration = (
                float(full["grid_duration_ns"])
                if isinstance(available, (bool, np.bool_)) and bool(available)
                else None
            )
            neighbors = (
                np.asarray(full["neighbors_ns"], dtype=float)
                if duration is not None
                else np.array([], dtype=float)
            )
            warnings = list(fit["warnings"])
            vector = np.array(
                [float(parameters[name]) for name in _DAMPED_PARAMETER_NAMES]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed local duration fit in row {index}: {error}"
            ) from error
        numeric = [ramp, reduced_chi2, rate, rate_error, *visibility, *vector]
        if duration is not None:
            numeric.append(duration)
        if (
            not np.isfinite(numeric).all()
            or ramp <= 0
            or rate_error <= 0
            or (
                duration is not None
                and (
                    duration < 2 * ramp
                    or not np.isfinite(neighbors).all()
                    or not np.any(np.isclose(neighbors, duration, atol=1e-9, rtol=0))
                    or not np.isclose(
                        duration / 2, round(duration / 2), atol=1e-9, rtol=0
                    )
                )
            )
        ):
            raise ValueError(
                f"Local duration fit row {index} has nonfinite or off-grid numbers"
            )
        if common_ramp is None:
            common_ramp = ramp
        elif ramp != common_ramp:
            raise ValueError("All five local duration fits must use the same ramp_ns")
        if not isinstance(fit["warnings"], list) or any(
            not isinstance(warning, str) for warning in warnings
        ):
            raise ValueError(f"Local duration fit row {index} warnings must be strings")
        unsupported = [
            warning for warning in warnings if warning != _TOLERATED_OPERATIONAL_WARNING
        ]
        row_reasons = []
        if not isinstance(available, (bool, np.bool_)):
            raise TypeError(f"Local duration fit row {index} has invalid availability")
        if not bool(available):
            row_reasons.append("full_candidate_unavailable")
        if not lo < rate < hi:
            row_reasons.append("rate_not_interior")
        if reduced_chi2 > limit:
            row_reasons.append("reduced_chi2_exceeds_limit")
        if visibility[0] < visibility_limit:
            row_reasons.append("visibility_00_below_minimum")
        if visibility[1] < visibility_limit:
            row_reasons.append("visibility_11_below_minimum")
        if unsupported:
            row_reasons.append("unsupported_fit_warning")
        prediction = None
        if duration is not None:
            prediction = damped_transfer(vector, [duration - 2 * ramp])[:, 0]
            if (
                not np.isfinite(prediction).all()
                or np.any(prediction < 0)
                or np.any(prediction > 1)
            ):
                raise ValueError(
                    f"Local duration fit row {index} predicts invalid probability"
                )
        diagnostic = dict(
            row_index=index,
            frequency_ghz=float(frequency),
            rate_mhz=rate,
            rate_local_standard_error_mhz=rate_error,
            reduced_chi2=reduced_chi2,
            visibility_00=visibility[0],
            visibility_11=visibility[1],
            full_duration_ns=duration,
            predicted_directional_probabilities=(
                prediction.tolist() if prediction is not None else None
            ),
            primary_score=float(np.min(prediction)) if prediction is not None else None,
            secondary_score=float(np.mean(prediction))
            if prediction is not None
            else None,
            warnings=warnings,
            reasons=row_reasons,
        )
        diagnostics.append(diagnostic)
        reasons.extend(f"row_{index}:{reason}" for reason in row_reasons)
        retained_warnings.extend(
            warning for warning in warnings if warning not in retained_warnings
        )

    passed = not reasons
    candidate = None
    if passed:
        selected = min(
            diagnostics,
            key=lambda row: (
                -row["primary_score"],
                -row["secondary_score"],
                round(abs(row["frequency_ghz"] - seed_frequency_ghz) * 1000, 12),
                row["frequency_ghz"],
            ),
        )
        candidate = {
            key: selected[key]
            for key in (
                "row_index",
                "frequency_ghz",
                "predicted_directional_probabilities",
                "primary_score",
                "secondary_score",
            )
        }
        candidate["duration_ns"] = selected["full_duration_ns"]
    return dict(
        candidate_generation_passed=passed,
        operational_response_candidate=candidate,
        rows=diagnostics,
        reasons=reasons,
        warnings=retained_warnings,
        row_models_all_warning_free=not retained_warnings,
        adoption_qualified=False,
        independent_holdout_required=True,
        ranking=(
            "minimum predicted direction, mean predicted direction, seed distance, lower frequency"
        ),
        claim=(
            "measured-row operational response candidate only; no resonance, gap, "
            "Hamiltonian profile, gate fidelity or adoption"
        ),
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
