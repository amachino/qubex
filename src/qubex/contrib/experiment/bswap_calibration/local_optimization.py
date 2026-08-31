"""Pure fixed-duration response-ridge and noise-aware local GP planning."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def _numeric(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _bounds(values: Sequence[float], name: str) -> tuple[float, float]:
    array = _numeric(values, name)
    if array.shape != (2,) or array[0] >= array[1]:
        raise ValueError(f"{name} must contain strictly increasing lower/upper bounds")
    return float(array[0]), float(array[1])


def _observations(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    scores: ArrayLike,
    shot_variances: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    arrays = tuple(
        _numeric(x, name)
        for x, name in zip(
            (amplitudes, frequencies_ghz, scores, shot_variances),
            ("amplitudes", "frequencies_ghz", "scores", "shot_variances"),
            strict=True,
        )
    )
    if (
        any(array.ndim != 1 for array in arrays)
        or len({len(array) for array in arrays}) != 1
    ):
        raise ValueError("Observations must be equally sized one-dimensional arrays")
    if len(arrays[0]) < 3 or np.any(arrays[3] <= 0):
        raise ValueError("Need at least three observations and positive shot variances")
    return arrays[0], arrays[1], arrays[2], arrays[3]


def ridge_scout_points(
    center_amplitude: float,
    center_frequency_ghz: float,
    *,
    amplitude_step: float = 0.0015,
    frequency_half_width_mhz: float = 0.6,
    frequency_points: int = 7,
    prior_slope_ghz_per_amplitude: float = 0.0,
    amplitude_bounds: Sequence[float] = (0.0, 0.99),
    frequency_bounds_ghz: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Plan three distinct amplitude rows around a prior response ridge.

    Parameters
    ----------
    center_amplitude : float
        Main command amplitude, not a Rabi frequency.
    center_frequency_ghz : float
        Prior fixed-duration score-peak frequency at center_amplitude.
    amplitude_step : float, optional
        Adjacent amplitude separation, default 0.0015. Reduce it when the
        ridge slope is unknown and the available frequency band is narrow.
    frequency_half_width_mhz : float, optional
        Per-row frequency half-width, default 0.6 MHz.
    frequency_points : int, optional
        Number of frequency samples per row, at least three.
    prior_slope_ghz_per_amplitude : float, optional
        Prior slope used only to center each row, default zero.
    amplitude_bounds, frequency_bounds_ghz : Sequence[float], optional
        Permitted physical controls. Frequency clipping never silently
        changes the planned band; insufficient coverage raises ValueError.

    Returns
    -------
    dict
        Physical amplitude/frequency pairs and explicit no-acquisition metadata.

    Notes
    -----
    This planner performs no measurements. At the upper amplitude bound the
    three rows are cap-2*step, cap-step, cap; they are never clipped duplicates.
    A prior slope is not a new measurement or a Hamiltonian resonance claim.
    """
    values = _numeric(
        [
            center_amplitude,
            center_frequency_ghz,
            amplitude_step,
            frequency_half_width_mhz,
            prior_slope_ghz_per_amplitude,
        ],
        "scout controls",
    )
    a0, f0, step, width, slope = values
    lower, upper = _bounds(amplitude_bounds, "amplitude_bounds")
    if not lower <= a0 <= upper or step <= 0 or 2 * step > upper - lower or width <= 0:
        raise ValueError(
            "The scout requires three distinct amplitudes within its bounds"
        )
    if (
        isinstance(frequency_points, bool)
        or int(frequency_points) != frequency_points
        or frequency_points < 3
    ):
        raise ValueError("frequency_points must be an integer >=3")
    first = np.clip(a0 - step, lower, upper - 2 * step)
    amplitudes = first + np.arange(3) * step
    rows, points = [], []
    for amplitude in amplitudes:
        center = f0 + slope * (amplitude - a0)
        frequencies = center + np.linspace(-width, width, frequency_points) / 1000
        if frequency_bounds_ghz is not None:
            low_f, high_f = _bounds(frequency_bounds_ghz, "frequency_bounds_ghz")
            if frequencies[0] < low_f or frequencies[-1] > high_f:
                raise ValueError(
                    "Requested scout band exceeds frequency bounds; revise the plan explicitly"
                )
        rows.append(
            {
                "amplitude": float(amplitude),
                "center_frequency_ghz": float(center),
                "frequencies_ghz": frequencies.tolist(),
            }
        )
        points.extend([[float(amplitude), float(f)] for f in frequencies])
    return {
        "points": points,
        "rows": rows,
        "requires_acquisition": True,
        "requested_shots": 0,
        "label": "fixed-duration response-ridge scout",
        "fixed_duration_and_ramp": True,
        "prior_slope_ghz_per_amplitude": float(slope),
    }


def _row_peak(
    frequencies: NDArray, scores: NDArray, variance: NDArray
) -> dict[str, Any]:
    order = np.argsort(frequencies)
    f, y, v = frequencies[order], scores[order], variance[order]
    if len(f) < 3 or np.any(np.diff(f) <= 0):
        return {
            "interior_peak": False,
            "reason": "need three distinct frequency samples",
        }
    index = int(np.argmax(y))
    if index in (0, len(f) - 1):
        direction = -1 if index == 0 else 1
        directional = direction * (y[-1] - y[0]) > 1.96 * np.sqrt(v[0] + v[-1])
        centers = (
            [float(f[index] + direction * np.ptp(f) / 2)]
            if directional
            else [float(f[0] - np.ptp(f) / 2), float(f[-1] + np.ptp(f) / 2)]
        )
        return {
            "interior_peak": False,
            "reason": "apparent peak is on the frequency boundary",
            "extension_centers_ghz": centers,
            "extension_direction_resolved": bool(directional),
        }
    local = slice(index - 1, index + 2)
    origin, scale = f[index], (f[index + 1] - f[index - 1]) / 2
    x = (f[local] - origin) / scale
    design = np.column_stack([np.ones(3), x, x * x])
    weighted = design / np.sqrt(v[local, None])
    covariance = np.linalg.inv(weighted.T @ weighted)
    coefficient = covariance @ (design.T @ (y[local] / v[local]))
    b, c = coefficient[1:]
    if c + 1.96 * np.sqrt(covariance[2, 2]) >= 0:
        return {
            "interior_peak": False,
            "reason": "negative local curvature is not resolved",
        }
    peak = -b / (2 * c)
    gradient = np.array([0.0, -1 / (2 * c), b / (2 * c * c)])
    error = float(scale * np.sqrt(max(gradient @ covariance @ gradient, 0)))
    frequency = float(origin + scale * peak)
    interval = [frequency - 1.96 * error, frequency + 1.96 * error]
    interior = bool(f[index - 1] < interval[0] < interval[1] < f[index + 1])
    return {
        "interior_peak": interior,
        "reason": "qualified local peak"
        if interior
        else "peak uncertainty reaches local bracket",
        "frequency_ghz": frequency,
        "standard_error_ghz": error,
        "ci95_ghz": interval,
        "local_frequency_bracket_ghz": [float(f[index - 1]), float(f[index + 1])],
        "uncertainty": "conditional three-point quadratic interpolation and supplied shot variance; not model-systematic coverage",
    }


def estimate_response_ridge(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    scores: ArrayLike,
    shot_variances: ArrayLike,
    *,
    reference_amplitude: float | None = None,
    frequency_bounds_ghz: Sequence[float] | None = None,
    maximum_reduced_chi2: float = 9.0,
) -> dict[str, Any]:
    """
    Estimate a local fixed-duration response ridge from measured scout rows.

    Parameters
    ----------
    amplitudes, frequencies_ghz, scores, shot_variances : ArrayLike
        Equal-length one-dimensional observations. Supply positive shot
        variances, including a finite-shot floor for extremal probabilities.
        At least three amplitude rows must have distinct frequency samples.
    reference_amplitude : float, optional
        Amplitude at which the fitted ridge frequency is reported.
    frequency_bounds_ghz : Sequence[float], optional
        Bound proposed follow-up frequency centers; no acquisition is performed.
    maximum_reduced_chi2 : float, optional
        Maximum shot-normalized residual of the local linear ridge, default 9.

    Returns
    -------
    dict
        Qualified flag, per-row peak intervals, ridge slope/covariance when
        supported, and explicit extension proposals when coverage is missing.

    Notes
    -----
    An interior raw maximum only locates three interpolation samples; it is
    not itself the selected peak. Require resolved downward curvature and an
    interior peak confidence interval. A fixed-duration score maximum is not
    necessarily an independently measured Hamiltonian resonance.
    """
    a, f, y, variance = _observations(
        amplitudes, frequencies_ghz, scores, shot_variances
    )
    rows, extensions = [], []
    for amplitude in np.unique(a):
        mask = a == amplitude
        row = {
            "amplitude": float(amplitude),
            **_row_peak(f[mask], y[mask], variance[mask]),
        }
        rows.append(row)
        for center in row.get("extension_centers_ghz", []):
            if frequency_bounds_ghz is not None:
                low, high = _bounds(frequency_bounds_ghz, "frequency_bounds_ghz")
                center = float(np.clip(center, low, high))
            extensions.append(
                {
                    "amplitude": float(amplitude),
                    "frequency_ghz": center,
                    "purpose": "extend measured peak bracket; not a selected resonance",
                }
            )
    result = {
        "qualified": False,
        "label": "fixed-duration response ridge",
        "rows": rows,
        "suggested_frequency_extensions": extensions,
        "requested_shots": 0,
    }
    if len(rows) < 3 or not all(row["interior_peak"] for row in rows):
        result["reason"] = (
            "Scout coverage or peak uncertainty is insufficient; acquire or refine the indicated rows"
        )
        return result
    a0 = float(np.mean(a) if reference_amplitude is None else reference_amplitude)
    if not np.isfinite(a0):
        raise ValueError("reference_amplitude must be finite numeric")
    centers = np.array([row["frequency_ghz"] for row in rows])
    errors = np.array([row["standard_error_ghz"] for row in rows])
    design = np.column_stack(
        [np.ones(len(rows)), np.array([r["amplitude"] for r in rows]) - a0]
    )
    weighted = design / errors[:, None]
    covariance = np.linalg.inv(weighted.T @ weighted)
    coefficient = covariance @ (design.T @ (centers / errors**2))
    residual = centers - design @ coefficient
    reduced = float(np.sum((residual / errors) ** 2) / max(len(rows) - 2, 1))
    result.update(
        qualified=bool(reduced <= maximum_reduced_chi2),
        reference_amplitude=a0,
        frequency_ghz=float(coefficient[0]),
        slope_ghz_per_amplitude=float(coefficient[1]),
        frequency_standard_error_ghz=float(np.sqrt(covariance[0, 0])),
        slope_standard_error=float(np.sqrt(covariance[1, 1])),
        covariance=covariance.tolist(),
        amplitude_coverage=[float(a.min()), float(a.max())],
        reduced_chi2=reduced,
        reason="local response ridge estimated"
        if reduced <= maximum_reduced_chi2
        else "linear ridge inconsistent with peak uncertainties",
        claim="local response coordinate model; not Hamiltonian resonance or gate fidelity",
    )
    return result


def propose_gp_point(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    scores: ArrayLike,
    shot_variances: ArrayLike,
    *,
    ridge: Mapping[str, Any],
    amplitude_bounds: Sequence[float],
    frequency_bounds_ghz: Sequence[float],
    frequency_half_width_mhz: float = 0.6,
    amplitude_scale: float = 0.0015,
    acquisition: Literal["ei", "ucb"] = "ei",
    exploration: float = 2.0,
    improvement_margin: float = 0.0,
    grid_size: tuple[int, int] = (31, 41),
    candidates: ArrayLike | None = None,
    allow_repeats: bool = False,
    include_anchor: bool = False,
    optimize_kernel: bool = True,
    seed: int = 781,
) -> dict[str, Any]:
    """
    Propose one bounded physical amplitude/frequency point from a noisy GP.

    Parameters
    ----------
    amplitudes, frequencies_ghz, scores, shot_variances : ArrayLike
        Existing observations and known positive shot variances.
    ridge : Mapping[str, Any]
        Qualified response-ridge estimate. Missing scout coverage must be
        resolved before using a GP, not hidden by a nonquadratic model.
    amplitude_bounds, frequency_bounds_ghz : Sequence[float]
        Explicit permitted physical-control intervals; never expanded here.
    frequency_half_width_mhz : float, optional
        Candidate residual-frequency band about the ridge, default 0.6 MHz.
    amplitude_scale : float, optional
        GP amplitude-coordinate scale in command units, default 0.0015.
    acquisition : {"ei", "ucb"}, optional
        Expected improvement or upper confidence bound, default EI.
    exploration, improvement_margin : float, optional
        UCB standard-deviation weight and EI score margin, respectively.
    grid_size : tuple[int, int], optional
        Cheap candidate-grid resolution, not a hardware acquisition budget.
    candidates : ArrayLike, optional
        Explicit physical (amplitude, GHz) candidates replacing the grid.
    allow_repeats : bool, optional
        Permit selecting measured coordinates, default False.
    include_anchor : bool, optional
        Return a separately marked optional repeatability anchor.
    optimize_kernel : bool, optional
        Fit bounded Matern hyperparameters with zero restarts, default True.
    seed : int, optional
        Deterministic GP seed.

    Returns
    -------
    dict
        One physical candidate, posterior statistics, a conservative observed
        incumbent, and optional anchor. No measurements or shots are requested.

    Notes
    -----
    Frequency coordinates subtract f0+slope*(A-A0), leaving amplitude and
    offset-from-ridge axes. Known noise is rescaled together with the scores.
    Rank the incumbent by posterior mean minus 1.96 standard deviations, not
    by the noisiest observed maximum. Posterior intervals are model dependent;
    a boundary proposal is not a resonance or optimum claim. The caller owns
    acquisition, anchor and independent-validation budgets, while duration
    and ramp remain fixed. Materialize a new pulse at the physical carrier
    using the caller's declared carrier-adaptive design policy.
    """
    a, f, y, noise = _observations(amplitudes, frequencies_ghz, scores, shot_variances)
    if not ridge.get("qualified", False):
        raise ValueError(
            "A qualified response ridge is required; first resolve missing scout coverage"
        )
    amin, amax = _bounds(amplitude_bounds, "amplitude_bounds")
    fmin, fmax = _bounds(frequency_bounds_ghz, "frequency_bounds_ghz")
    if not 0 <= amin < amax <= 1:
        raise ValueError("Command amplitude bounds must lie within [0,1]")
    controls = _numeric(
        [
            ridge["reference_amplitude"],
            ridge["frequency_ghz"],
            ridge["slope_ghz_per_amplitude"],
            amplitude_scale,
            frequency_half_width_mhz,
            exploration,
            improvement_margin,
        ],
        "GP controls",
    )
    a0, f0, slope, a_scale, width_mhz, beta, margin = controls
    width = width_mhz / 1000
    if (
        a_scale <= 0
        or width <= 0
        or beta < 0
        or margin < 0
        or acquisition not in ("ei", "ucb")
    ):
        raise ValueError("Invalid GP scales or acquisition settings")

    def coordinates(points: NDArray) -> NDArray:
        return np.column_stack(
            [
                (points[:, 0] - a0) / a_scale,
                (points[:, 1] - f0 - slope * (points[:, 0] - a0)) / width,
            ]
        )

    observed = np.column_stack([a, f])
    if candidates is None:
        if len(grid_size) != 2 or any(
            isinstance(n, bool) or int(n) != n or n < 2 for n in grid_size
        ):
            raise ValueError("grid_size requires two integer dimensions >=2")
        aa, offsets = np.meshgrid(
            np.linspace(amin, amax, grid_size[0]),
            np.linspace(-width, width, grid_size[1]),
            indexing="ij",
        )
        proposed = np.column_stack(
            [aa.ravel(), (f0 + slope * (aa - a0) + offsets).ravel()]
        )
    else:
        proposed = _numeric(candidates, "candidates")
        if proposed.ndim != 2 or proposed.shape[1] != 2:
            raise ValueError("candidates must have shape (N,2)")
    admitted = (
        (proposed[:, 0] >= amin)
        & (proposed[:, 0] <= amax)
        & (proposed[:, 1] >= fmin)
        & (proposed[:, 1] <= fmax)
        & (abs(coordinates(proposed)[:, 1]) <= 1 + 1e-10)
    )
    proposed = proposed[admitted]
    repeats = np.any(
        (abs(proposed[:, None, 0] - a[None, :]) < 1e-10)
        & (abs(proposed[:, None, 1] - f[None, :]) < 1e-12),
        axis=1,
    )
    if not allow_repeats:
        proposed, repeats = proposed[~repeats], repeats[~repeats]
    if len(proposed) == 0:
        raise ValueError(
            "No permitted unmeasured candidates remain; revise bounds or request repeats explicitly"
        )
    precision = noise.min() / noise
    weights = precision / precision.sum()
    location = float(weights @ y)
    scale = max(float(np.sqrt(weights @ ((y - location) ** 2))), 1e-6)
    # sklearn's inferred scalar defaults understate its supported vector
    # length_scale/alpha arguments and optimizer=None; keep casts local.
    kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
        length_scale=cast(Any, [1.0, 0.3]), length_scale_bounds=(0.05, 8.0), nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=cast(Any, noise / scale**2 + 1e-10),
        optimizer=cast(Any, "fmin_l_bfgs_b" if optimize_kernel else None),
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(coordinates(observed), (y - location) / scale)
    mean, std = cast(
        tuple[NDArray, NDArray], gp.predict(coordinates(proposed), return_std=True)
    )
    mean, std = location + scale * mean, scale * std
    measured_mean, measured_std = cast(
        tuple[NDArray, NDArray], gp.predict(coordinates(observed), return_std=True)
    )
    measured_mean, measured_std = location + scale * measured_mean, scale * measured_std
    eligible = (a >= amin) & (a <= amax) & (f >= fmin) & (f <= fmax)
    if not eligible.any():
        raise ValueError("No observed incumbent lies within the permitted bounds")
    conservative = np.where(eligible, measured_mean - 1.96 * measured_std, -np.inf)
    best = int(np.argmax(conservative))
    if acquisition == "ucb":
        utility = mean + beta * std
    else:
        improvement = mean - measured_mean[best] - margin
        z = improvement / np.maximum(std, 1e-15)
        utility = improvement * norm.cdf(z) + std * norm.pdf(z)
    selected = int(np.argmax(utility))
    point = proposed[selected]
    incumbent = {
        "index": best,
        "amplitude": float(a[best]),
        "frequency_ghz": float(f[best]),
        "observed_score": float(y[best]),
        "posterior_mean": float(measured_mean[best]),
        "posterior_standard_deviation": float(measured_std[best]),
        "ranking_score": float(conservative[best]),
    }
    candidate = {
        "amplitude": float(point[0]),
        "frequency_ghz": float(point[1]),
        "posterior_mean": float(mean[selected]),
        "posterior_standard_deviation": float(std[selected]),
        "acquisition_value": float(utility[selected]),
        "previously_measured": bool(repeats[selected]),
        "at_boundary": bool(
            np.isclose(point[0], [amin, amax], atol=1e-12, rtol=0).any()
            or np.isclose(point[1], [fmin, fmax], atol=1e-12, rtol=0).any()
            or abs(coordinates(point[None, :])[0, 1]) >= 1 - 1e-10
        ),
    }
    anchor = (
        None
        if not include_anchor
        else {
            "amplitude": incumbent["amplitude"],
            "frequency_ghz": incumbent["frequency_ghz"],
            "purpose": "repeatability anchor, not independent validation",
        }
    )
    return {
        "candidate": candidate,
        "best_observed": incumbent,
        "optional_anchor": anchor,
        "acquisition": acquisition,
        "kernel": str(gp.kernel_),
        "fit_warnings": [str(w.message) for w in caught],
        "requested_shots": 0,
        "validation_required": True,
        "fixed_duration_and_ramp": True,
        "coordinate_model": "amplitude and offset from fixed-duration response ridge",
        "budget_scope": "proposal only; acquisition, anchors and fresh validation are separate caller budgets",
        "uncertainty_scope": "conditional on the frozen ridge and fitted kernel; supplied shot noise included, ridge/model systematics excluded",
    }


def plan_frequency_extensions(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    scores: ArrayLike,
    shot_variances: ArrayLike,
    *,
    ridge: Mapping[str, Any],
    frequency_bounds_ghz: Sequence[float],
    round_index: int,
    max_rounds: int = 2,
    frequency_points: int = 7,
    frequency_half_width_mhz: float = 0.6,
) -> dict[str, Any]:
    """
    Plan bounded additional frequencies only for unqualified scout rows.

    Parameters
    ----------
    amplitudes, frequencies_ghz, scores, shot_variances : ArrayLike
        Existing physical observations and positive shot variances.
    ridge : Mapping[str, Any]
        Result from estimate_response_ridge for those observations.
    frequency_bounds_ghz : Sequence[float]
        Hard physical frequency limits, supplied by the caller.
    round_index : int
        Zero-based extension round. Index max_rounds returns no points.
    max_rounds : int, optional
        Maximum caller-controlled extension rounds, default two.
    frequency_points : int, optional
        Candidate points per extension interval, default seven.
    frequency_half_width_mhz : float, optional
        Half-width around proposed row centers, default 0.6 MHz.

    Returns
    -------
    dict
        Deduplicated physical points, per-row reasons, and exhaustion status.
        No measurement is made and no failed ridge becomes qualified.

    Notes
    -----
    Boundary rows follow the estimator's indicated directions. A shallow
    unresolved interior response may request symmetric outward coverage,
    without inventing a peak direction. A local peak with excessive uncertainty
    receives denser samples inside its existing bracket. Previously measured
    coordinates and qualified rows are never scheduled again by this helper.
    """
    a, f, y, variance = _observations(
        amplitudes, frequencies_ghz, scores, shot_variances
    )
    low, high = _bounds(frequency_bounds_ghz, "frequency_bounds_ghz")
    for value, name in ((round_index, "round_index"), (max_rounds, "max_rounds")):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if (
        isinstance(frequency_points, bool)
        or int(frequency_points) != frequency_points
        or frequency_points < 3
    ):
        raise ValueError("frequency_points must be an integer >=3")
    width = float(_numeric(frequency_half_width_mhz, "frequency_half_width_mhz")) / 1000
    if width <= 0:
        raise ValueError("frequency_half_width_mhz must be positive")
    result: dict[str, Any] = {
        "points": [],
        "rows": [],
        "round_index": int(round_index),
        "maximum_rounds": int(max_rounds),
        "exhausted": bool(round_index >= max_rounds),
        "qualified_ridge": bool(ridge.get("qualified", False)),
        "requested_shots": 0,
        "requires_acquisition": False,
        "validation_required": True,
    }
    if result["exhausted"]:
        result["reason"] = "Declared extension-round budget exhausted"
        return result
    if result["qualified_ridge"]:
        result["reason"] = (
            "The response ridge is already qualified; no extension needed"
        )
        return result
    seen: set[tuple[float, float]] = set()
    for row in ridge.get("rows", []):
        if row.get("interior_peak", False):
            continue
        amplitude = float(row["amplitude"])
        mask = abs(a - amplitude) < 1e-10
        if mask.sum() < 3:
            continue
        row_f, row_y, row_v = f[mask], y[mask], variance[mask]
        centers = list(row.get("extension_centers_ghz", []))
        if not centers:
            centers = [
                item["frequency_ghz"]
                for item in ridge.get("suggested_frequency_extensions", [])
                if abs(float(item["amplitude"]) - amplitude) < 1e-10
            ]
        grids = []
        purpose = "extend measured boundary coverage"
        if centers:
            grids = [
                np.linspace(
                    float(center) - width, float(center) + width, frequency_points
                )
                for center in centers
            ]
        elif "local_frequency_bracket_ghz" in row:
            left, right = row["local_frequency_bracket_ghz"]
            grids = [np.linspace(left, right, frequency_points)]
            purpose = "refine uncertain local peak bracket"
        elif "curvature" in row.get("reason", ""):
            if np.ptp(row_y) <= 3 * np.sqrt(2 * np.max(row_v)):
                grids = [
                    np.linspace(row_f.min() - 2 * width, row_f.min(), frequency_points),
                    np.linspace(row_f.max(), row_f.max() + 2 * width, frequency_points),
                ]
                purpose = "symmetric coverage; curvature unresolved"
        added = []
        for grid in grids:
            for frequency in grid:
                if not low <= frequency <= high:
                    continue
                point = (amplitude, float(frequency))
                key = (round(point[0], 12), round(point[1], 12))
                measured = np.any(
                    (abs(a - point[0]) < 1e-10) & (abs(f - point[1]) < 1e-12)
                )
                if key in seen or measured:
                    continue
                seen.add(key)
                added.append(list(point))
                result["points"].append(list(point))
        result["rows"].append(
            {
                "amplitude": amplitude,
                "purpose": purpose
                if grids
                else "no supported extension; inspect precision/coverage",
                "points": added,
                "previous_reason": row.get("reason"),
            }
        )
    result["requires_acquisition"] = bool(result["points"])
    result["exhausted"] = not bool(result["points"])
    result["reason"] = (
        "Acquire these bounded diagnostic points, then re-estimate the response ridge"
        if result["points"]
        else "No new supported points remain within the permitted frequency bounds"
    )
    return result


def select_conservative_plateau(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    scores: ArrayLike,
    shot_variances: ArrayLike,
    *,
    gate_kind: str,
    seed_amplitude: float,
    seed_frequency_ghz: float,
    minimum_lower_confidence: float = 0.85,
    frequency_neighborhood_mhz: float = 0.6,
    significance: float = 0.05,
) -> dict[str, Any]:
    """
    Retain an independent root seed on a high, unresolved coherence plateau.

    Parameters
    ----------
    amplitudes, frequencies_ghz, scores, shot_variances : ArrayLike
        Supplied raw root population-plus-coherence scores and their positive
        shot variances, not population-only or clipped/mitigated scores.
    gate_kind : str
        Only sqrt_bswap admits this fallback. Full bSWAP is rejected.
    seed_amplitude, seed_frequency_ghz : float
        Independently chosen chevron/duration seed controls. Keep the supplied
        frequency exactly and use the nearest measured amplitude row.
    minimum_lower_confidence : float, optional
        Minimum simultaneous normal-shot lower bound on every selected score,
        default 0.85. This is a score criterion, not gate fidelity.
    frequency_neighborhood_mhz : float, optional
        Predeclared neighborhood around the seed, default 0.6 MHz.
    significance : float, optional
        Simultaneous interval error rate and constant-response test level,
        default 0.05.

    Returns
    -------
    dict
        Accepted flag, retained candidate or None, and explicit reasons.
        qualified_ridge and gp_allowed remain False even when accepted.

    Notes
    -----
    At least three distinct frequencies must bracket the independent seed.
    Bonferroni normal intervals account for supplied finite-shot variance;
    a weighted constant-response chi-square test must not resolve variation.
    Failure to detect variation is not proof of an exactly flat response.
    New independent shots must validate the candidate before later stages.
    This helper performs no acquisition, picks no observed score maximum,
    and does not estimate a Hamiltonian resonance or authorize GP refinement.
    """
    a, f, y, variance = _observations(
        amplitudes, frequencies_ghz, scores, shot_variances
    )
    controls = _numeric(
        [
            seed_amplitude,
            seed_frequency_ghz,
            minimum_lower_confidence,
            frequency_neighborhood_mhz,
            significance,
        ],
        "plateau controls",
    )
    seed_a, seed_f, threshold, width_mhz, alpha = controls
    if not 0 < alpha < 1 or not 0 < threshold <= 1 or width_mhz <= 0:
        raise ValueError(
            "Invalid plateau confidence threshold or frequency neighborhood"
        )
    result: dict[str, Any] = {
        "accepted": False,
        "candidate": None,
        "reasons": [],
        "qualified_ridge": False,
        "gp_allowed": False,
        "validation_required": True,
        "requested_shots": 0,
        "mode": "conservative root plateau; independent seed retained",
    }
    if gate_kind != "sqrt_bswap":
        result["reasons"] = ["This fallback is restricted to raw root coherence scores"]
        return result
    levels = np.unique(a)
    amplitude = float(min(levels, key=lambda value: (abs(value - seed_a), value)))
    selected = (abs(a - amplitude) < 1e-10) & (
        abs(f - seed_f) <= width_mhz / 1000 + 1e-12
    )
    indices = np.flatnonzero(selected)
    row_f, row_y, row_v = f[selected], y[selected], variance[selected]
    result["selected_indices"] = indices.tolist()
    result["nearest_seed_amplitude"] = amplitude
    if len(np.unique(row_f)) < 3 or not row_f.min() < seed_f < row_f.max():
        result["reasons"] = [
            "Need three distinct frequencies bracketing the independent seed"
        ]
        return result
    z = float(norm.ppf(1 - alpha / (2 * len(row_y))))
    lower = row_y - z * np.sqrt(row_v)
    precision = row_v.min() / row_v
    weighted_mean = float(precision @ row_y / precision.sum())
    statistic = float(np.sum((row_y - weighted_mean) ** 2 / row_v))
    p_value = float(chi2.sf(statistic, len(row_y) - 1))
    result.update(
        minimum_simultaneous_lower_bound=float(lower.min()),
        weighted_mean_score=weighted_mean,
        constant_response_chi2=statistic,
        constant_response_degrees_of_freedom=len(row_y) - 1,
        variation_test_p_value=p_value,
        significance=float(alpha),
        confidence_scope="simultaneous normal approximation using supplied shot variances; SPAM/model systematics excluded",
        flatness_claim="variation unresolved, not proven absent",
    )
    reasons = []
    if lower.min() < threshold:
        reasons.append(
            "Some nearby scores lack the required high lower confidence bound"
        )
    if np.any(lower > 1.0):
        reasons.append(
            "A score is confidently above its physical range; inspect the measurement model"
        )
    if p_value < alpha:
        reasons.append("Frequency-dependent score variation is statistically resolved")
    if reasons:
        result["reasons"] = reasons
        return result
    result.update(
        accepted=True,
        candidate={"amplitude": amplitude, "frequency_ghz": float(seed_f)},
        reasons=[
            "High nearby raw coherence scores with unresolved variation; retain the independent seed without GP"
        ],
    )
    return result
