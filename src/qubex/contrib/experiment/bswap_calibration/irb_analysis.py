"""Raw all-shot paired-seed full-bSWAP IRB analysis; no hardware access."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit


def decay(
    depth: ArrayLike, amplitude: float, p: float, floor: float
) -> NDArray[np.float64]:
    """Return the standard RB survival model."""
    return amplitude * p ** np.asarray(depth, dtype=float) + floor


def _validated_decay_inputs(
    depths: ArrayLike, matrix: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate all raw controls before selecting positive fit depths."""
    depths = np.asarray(depths, dtype=float)
    if (
        depths.ndim != 1
        or not np.isfinite(depths).all()
        or np.any(depths < 0)
        or np.any(depths != np.floor(depths))
        or len(np.unique(depths)) != len(depths)
    ):
        raise ValueError("Depths must be finite, nonnegative, unique integers")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(depths) or matrix.shape[0] < 2:
        raise ValueError("Need a (seed, depth) matrix with at least two seeds")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(
            "Raw all-shot probabilities, including depth-zero controls, must be finite and in [0, 1]"
        )
    return depths, matrix


def fit_decay(
    depths: ArrayLike, matrix: ArrayLike, *, initial: Sequence[float] | None = None
) -> dict[str, Any]:
    """Fit across seeds and expose identifiability rather than only R-squared."""
    all_x, matrix = _validated_decay_inputs(depths, matrix)
    all_mean = matrix.mean(axis=0)
    all_sem = np.maximum(matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix)), 0.002)
    # At m=0 the cache emits no inverse pulses. For m>=1 the random inverse
    # has a nonzero mean error absorbed in A/B; m=0 is therefore a SPAM control.
    used = all_x > 0
    x, mean, sem = all_x[used], all_mean[used], all_sem[used]
    if len(x) < 4:
        raise ValueError("At least four positive depths are required")
    starts = (
        [initial]
        if initial is not None
        else [[max(0.1, mean[0] - 0.25), p, 0.25] for p in (0.65, 0.85, 0.95, 0.99)]
    )
    candidates = []
    for start in starts:
        try:
            pars, covariance = curve_fit(
                decay,
                x,
                mean,
                p0=start,
                sigma=sem,
                absolute_sigma=True,
                bounds=([0, 1e-6, 0], [1.2, 0.999999999, 0.8]),
                maxfev=10000,
            )
            residual = mean - decay(x, *pars)
            candidates.append((float(np.sum((residual / sem) ** 2)), pars, covariance))
        except (RuntimeError, ValueError, FloatingPointError):  # noqa: PERF203 - Independent fit seeds can fail.
            continue
    if not candidates:
        raise RuntimeError("No finite bounded RB fit")
    chi2, pars, covariance = min(candidates, key=lambda item: item[0])
    prediction = decay(x, *pars)
    errors = np.sqrt(np.maximum(0, np.diag(covariance)))
    total = float(np.sum((mean - mean.mean()) ** 2))
    r2 = 1 - float(np.sum((mean - prediction) ** 2)) / total if total else float("nan")
    amplitude, p, floor = map(float, pars)
    reasons = []
    if not np.isfinite(covariance).all() or not np.isfinite(r2):
        reasons.append("nonfinite_fit")
    if r2 < 0.90:
        reasons.append("nonexponential_or_noisy_mean")
    if amplitude < 0.30:
        reasons.append("low_contrast")
    if errors[0] > 0.10 or errors[2] > 0.10 or errors[1] > 0.03:
        reasons.append("poorly_identified_parameters")
    if p ** max(x) > 0.10:
        reasons.append("range_does_not_reach_floor_region")
    if p > 0.9999 or floor > 0.79:
        reasons.append("parameter_at_upper_bound")
    if chi2 / max(1, len(x) - 3) > 8:
        reasons.append("large_weighted_residual")
    return {
        "amplitude": amplitude,
        "p": p,
        "floor": floor,
        "errors": dict(
            zip(("amplitude", "p", "floor"), map(float, errors), strict=True)
        ),
        "covariance": covariance.tolist(),
        "r2": r2,
        "reduced_chi2": chi2 / max(1, len(x) - 3),
        "remaining_contrast_fraction": float(p ** max(x)),
        "means": all_mean.tolist(),
        "sem": all_sem.tolist(),
        "fit_depths": x.tolist(),
        "depth_zero_role": "empty-circuit control, not fitted",
        "fit_quality_pass": not reasons,
        "reasons": reasons,
    }


def analyze_irb(
    depths: ArrayLike,
    reference: ArrayLike,
    interleaved: ArrayLike,
    *,
    bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Return a conditional IRB estimate only when both decay fits are usable.

    Parameters
    ----------
    depths : array_like
        Distinct nonnegative integer Clifford depths; zero is a control only.
    reference, interleaved : array_like
        Matched raw survival arrays with shape (seed, depth), each in [0, 1].
    bootstrap : int, default 300
        Positive number of paired-seed resamples.
    seed : int, default 42
        Bootstrap random seed, separate from the acquired circuit seeds.

    Returns
    -------
    dict
        Fit diagnostics and rejection reasons. A fidelity estimate is present
        only when both fits and resampling are usable; its interval is statistical,
        not a reference-noise systematic bound.
    """
    if (
        isinstance(bootstrap, (bool, np.bool_))
        or not isinstance(bootstrap, (int, np.integer))
        or bootstrap <= 0
    ):
        raise ValueError("bootstrap must be a positive integer")
    bootstrap = int(bootstrap)
    depths, reference = _validated_decay_inputs(depths, reference)
    _, interleaved = _validated_decay_inputs(depths, interleaved)
    if (
        reference.shape != interleaved.shape
        or reference.ndim != 2
        or reference.shape[1] != len(depths)
    ):
        raise ValueError("Need matched (seed, depth) reference/interleaved arrays")
    if (
        reference.shape[0] < 4
        or not np.isfinite(reference).all()
        or not np.isfinite(interleaved).all()
    ):
        raise ValueError(
            "Need at least four complete paired seeds; no silent missing-point selection"
        )
    fits = {
        "reference": fit_decay(depths, reference),
        "interleaved": fit_decay(depths, interleaved),
    }
    diagnostic = float(0.25 + 0.75 * fits["interleaved"]["p"] / fits["reference"]["p"])
    reasons = [
        f"{mode}:{reason}" for mode, fit in fits.items() for reason in fit["reasons"]
    ]
    if not 0.25 <= diagnostic <= 1:
        reasons.append("nonphysical_irb_estimate")
    draws = []
    if not reasons:
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            indices = rng.integers(0, len(reference), len(reference))
            try:
                ref_fit = fit_decay(
                    depths,
                    reference[indices],
                    initial=[fits["reference"][k] for k in ("amplitude", "p", "floor")],
                )
                irb_fit = fit_decay(
                    depths,
                    interleaved[indices],
                    initial=[
                        fits["interleaved"][k] for k in ("amplitude", "p", "floor")
                    ],
                )
                if not (ref_fit["fit_quality_pass"] and irb_fit["fit_quality_pass"]):
                    continue
                draw = 0.25 + 0.75 * irb_fit["p"] / ref_fit["p"]
                if np.isfinite(draw):
                    draws.append(draw)
            except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
                continue
        if len(draws) < 0.9 * bootstrap:
            reasons.append("bootstrap_instability")
    interval = np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) >= 2 else None
    if interval is None or not np.isfinite(interval).all():
        interval = None
        reasons.append("bootstrap_interval_unavailable")
    if interval is not None and interval[1] - interval[0] > 0.1:
        reasons.append("large_statistical_uncertainty")
    return dict(
        fits=fits,
        fidelity_estimate=diagnostic if not reasons else None,
        diagnostic_estimate_unclipped=diagnostic,
        statistical_interval_95=interval,
        quote_as_irb_estimate=not reasons,
        reasons=reasons,
        bootstrap_successes=len(draws),
        bootstrap_seed=seed,
        bootstrap_requested=bootstrap,
        target="ideal plus-i full bSWAP; physical residual ZZ counts as error",
        claim="all-shot raw-classified conditional IRB estimate; bootstrap covers seed statistics, not gate-dependent/reference-noise systematic bias",
    )
