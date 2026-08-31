"""Pure uniform-depth IRB planning, paired acquisition costs and final-tail checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, floor, log
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value: float) -> float:
    p = float(value)
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError(
            "Planning decay probabilities must be strictly between zero and one"
        )
    return p


def _uniform_depths(depths: Sequence[int] | ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(depths, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 5
        or not np.isfinite(values).all()
        or values[0] != 0
        or np.any(values != np.floor(values))
    ):
        raise ValueError(
            "Require uniform integer depths with zero and at least four positive points"
        )
    differences = np.diff(values)
    if differences[0] <= 0 or not np.all(differences == differences[0]):
        raise ValueError("Depths must be one increasing uniform integer grid")
    return values.astype(np.int64)


def plan_uniform_irb_depths(
    reference_p: float,
    interleaved_p: float,
    *,
    source_kind: Literal["qualified_pilot", "manual_prior"],
    source_label: str,
    reference_qualified: bool = False,
    interleaved_qualified: bool = False,
    reference_p_bounds: tuple[float, float] | None = None,
    interleaved_p_bounds: tuple[float, float] | None = None,
    target_positive_depths: int = 40,
    min_positive_depths: int = 20,
    preferred_max_positive_depths: int = 60,
    samples_per_fast_decay: int = 4,
    tail_decay_lengths: float = 5.0,
    tail_fraction: float = 0.01,
) -> dict[str, Any]:
    """
    Plan one linear integer Clifford grid resolving both IRB decays to their tails.

    Parameters
    ----------
    reference_p, interleaved_p : float
        Independent-pilot decay rates strictly between zero and one.
    source_kind : {"qualified_pilot", "manual_prior"}
        Explicit source. Failed pilots cannot silently become manual priors.
    source_label : str
        Inspectable pilot artifact or explicitly supplied manual-prior provenance.
    reference_qualified, interleaved_qualified : bool, optional
        Both must be true for qualified-pilot planning, default false.
    reference_p_bounds, interleaved_p_bounds : tuple[float, float], optional
        Pilot uncertainty intervals containing each rate and strictly inside
        (0,1). Lower bounds control resolution; upper bounds control tail extent.
    target_positive_depths, min_positive_depths : int, optional
        Preferred size and minimum positive fit points, default 40 and 20.
    preferred_max_positive_depths : int, optional
        Warning threshold only, default 60. Required points are never truncated.
    samples_per_fast_decay : int, optional
        At least four integer-grid samples per fastest e-fold, default four.
    tail_decay_lengths : float, optional
        At least this many slow e-folds, default five.
    tail_fraction : float, optional
        Maximum predicted fraction of initial contrast at the end, default 0.01.

    Returns
    -------
    dict[str, Any]
        Common REF/IRB depths including zero, positive fit depths, step, tail
        predictions and source provenance. This is planning, not qualification.

    Raises
    ------
    ValueError
        Unqualified source, unbounded decay, invalid controls or resolution
        impossible on integer Clifford depth. Impractically large grids are
        explicitly refused, never shortened before the required tail.

    Notes
    -----
    L=-1/log(p). Step is no larger than L_fast/samples_per_fast_decay and the
    maximum depth covers max(tail_decay_lengths,-log(tail_fraction))*L_slow.
    Zero is an empty-circuit SPAM control, not a fitted random-inverse depth.
    Final data still require their own decay and measured-tail assessment.
    No clock, random seed, files, hardware or calibration state is accessed.
    """
    rates = [_probability(reference_p), _probability(interleaved_p)]
    if (
        source_kind not in ("qualified_pilot", "manual_prior")
        or not isinstance(source_label, str)
        or not source_label.strip()
    ):
        raise ValueError(
            "An explicit qualified pilot or manual-prior source is required"
        )
    if source_kind == "qualified_pilot" and not (
        isinstance(reference_qualified, (bool, np.bool_))
        and bool(reference_qualified)
        and isinstance(interleaved_qualified, (bool, np.bool_))
        and bool(interleaved_qualified)
    ):
        raise ValueError(
            "Both pilot fits must be qualified; obtain a new pilot or supply an explicit manual prior"
        )
    minimum = _integer(min_positive_depths, "min_positive_depths", 4)
    target = _integer(target_positive_depths, "target_positive_depths", minimum)
    preferred_max = _integer(
        preferred_max_positive_depths, "preferred_max_positive_depths", target
    )
    resolution = _integer(samples_per_fast_decay, "samples_per_fast_decay", 4)
    tail = _finite(tail_fraction, "tail_fraction")
    lengths = _finite(tail_decay_lengths, "tail_decay_lengths")
    if not 0 < tail < 1 or lengths <= 0:
        raise ValueError(
            "Require tail_fraction in (0,1) and positive tail_decay_lengths"
        )
    intervals = []
    for p, interval in zip(
        rates, (reference_p_bounds, interleaved_p_bounds), strict=True
    ):
        low, high = (
            (p, p)
            if interval is None
            else tuple(_probability(value) for value in interval)
        )
        if not low <= p <= high:
            raise ValueError("Each pilot interval must contain its central decay rate")
        intervals.append((low, high))
    fast = -1 / log(min(interval[0] for interval in intervals))
    slow = -1 / log(max(interval[1] for interval in intervals))
    maximum_step = floor(fast / resolution)
    if maximum_step < 1:
        raise ValueError(
            "Fast decay cannot have four samples per e-fold on integer Clifford depths"
        )
    required_depth = ceil(max(lengths, -log(tail)) * slow)
    step = min(maximum_step, max(1, ceil(required_depth / target)))
    count = max(minimum, ceil(required_depth / step))
    maximum_depth = count * step
    if count > 100_000 or maximum_depth > np.iinfo(np.int64).max:
        raise ValueError(
            f"Full uniform design requires {count} positive points through depth {maximum_depth}; too large to materialize, not truncated"
        )
    depths = list(range(0, maximum_depth + step, step))
    warnings = []
    if count > preferred_max:
        warnings.append(
            "Required fast-decay resolution exceeds the preferred depth count; no tail truncation"
        )
    if source_kind == "manual_prior":
        warnings.append("Manual planning prior is not an independently qualified pilot")
    return dict(
        depths=depths,
        fit_depths=depths[1:],
        uniform_step=step,
        maximum_depth=maximum_depth,
        positive_depth_count=count,
        fast_decay_length=fast,
        slow_decay_length=slow,
        samples_per_fast_decay=resolution,
        tail_decay_lengths=max(lengths, -log(tail)),
        tail_fraction_target=tail,
        predicted_tail_reference=float(rates[0] ** maximum_depth),
        predicted_tail_interleaved=float(rates[1] ** maximum_depth),
        upper_tail_reference=float(intervals[0][1] ** maximum_depth),
        upper_tail_interleaved=float(intervals[1][1] ** maximum_depth),
        source_kind=source_kind,
        source_label=source_label,
        reference_p=rates[0],
        interleaved_p=rates[1],
        pilot_intervals=intervals,
        uses_pilot_bounds=reference_p_bounds is not None
        or interleaved_p_bounds is not None,
        depth_zero_role="empty-circuit control, not fitted",
        pairing="one common depth grid and identical seed identities for reference/interleaved",
        requires_independent_tail_validation=True,
        warnings=warnings,
    )


def estimate_irb_cost(
    depths: Sequence[int] | ArrayLike,
    seeds: Sequence[int],
    shots: int,
    *,
    shot_interval_ns: float = 1_000_000.0,
    circuit_durations_ns: ArrayLike | None = None,
    per_acquisition_setup_seconds: float | None = None,
    maximum_shots: int | None = None,
    maximum_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Account for every paired IRB circuit without truncating an over-budget design.

    Parameters
    ----------
    depths, seeds : Sequence[int]
        Full uniform depth grid including zero and at least four unique seed IDs.
        Seed order is preserved in both reference and interleaved conditions.
    shots : int
        Positive shot count per circuit in each condition.
    shot_interval_ns : float, optional
        Trailing idle per shot, default 1 ms. It excludes circuit execution.
    circuit_durations_ns : ArrayLike, optional
        Nonnegative durations shaped (2, number of seeds, number of depths),
        ordered reference/interleaved. Include complete executed shot duration,
        including readout, or document any unmodelled contributions externally.
    per_acquisition_setup_seconds : float, optional
        Additional upload/setup/processing time per call, excluding shot runtime.
        Omission is unknown overhead, not zero. Explicit zero is an assumption.
    maximum_shots : int, optional
        Requested-shot budget for this entire paired run.
    maximum_wall_seconds : float, optional
        Available wall time. An incomplete estimate cannot certify this budget.

    Returns
    -------
    dict[str, Any]
        Counts, idle and pulse costs, wall estimate, and explicit budget status.
        Unknown wall components remain None; depths/seeds are never altered.

    Notes
    -----
    Circuit durations are caller-supplied compiled/measured evidence, not a
    constant invented per Clifford. Actual timing can exceed this estimate.
    No acquisition order, ownership, reservation or calibration is changed.
    """
    grid = _uniform_depths(depths)
    identities = [_integer(seed, "seed", 0) for seed in seeds]
    if len(identities) < 4 or len(set(identities)) != len(identities):
        raise ValueError("IRB needs at least four unique paired seeds")
    nshots = _integer(shots, "shots")
    interval = _finite(shot_interval_ns, "shot_interval_ns")
    if interval < 0:
        raise ValueError("shot_interval_ns must be nonnegative")
    circuits = 2 * len(identities) * len(grid)
    total = circuits * nshots
    idle = total * interval * 1e-9
    sequence = None
    if circuit_durations_ns is not None:
        durations = np.asarray(circuit_durations_ns, dtype=float)
        if (
            durations.shape != (2, len(identities), len(grid))
            or not np.isfinite(durations).all()
            or np.any(durations < 0)
        ):
            raise ValueError(
                "Circuit duration shape must be (2,seeds,depths), finite and nonnegative"
            )
        sequence = float(durations.sum() * nshots * 1e-9)
    setup = None
    if per_acquisition_setup_seconds is not None:
        overhead = _finite(
            per_acquisition_setup_seconds, "per_acquisition_setup_seconds"
        )
        if overhead < 0:
            raise ValueError("Per-acquisition setup time must be nonnegative")
        setup = circuits * overhead
    lower_bound = (
        idle
        + (0.0 if sequence is None else sequence)
        + (0.0 if setup is None else setup)
    )
    wall = lower_bound if sequence is not None and setup is not None else None
    shot_ok = (
        None
        if maximum_shots is None
        else total <= _integer(maximum_shots, "maximum_shots")
    )
    time_ok = None
    if maximum_wall_seconds is not None:
        seconds = _finite(maximum_wall_seconds, "maximum_wall_seconds")
        if seconds <= 0:
            raise ValueError("maximum_wall_seconds must be positive")
        if lower_bound > seconds:
            time_ok = False
        elif wall is not None:
            time_ok = wall <= seconds
    status = "not_specified"
    reasons = []
    if maximum_shots is not None or maximum_wall_seconds is not None:
        status = "within_declared_budgets"
    if maximum_wall_seconds is not None and time_ok is None:
        status = "incomplete_time_estimate"
        reasons.append(
            "Circuit or setup duration is unknown; idle-only cost cannot certify wall budget"
        )
    if shot_ok is False or time_ok is False:
        status = "exceeds_declared_budget"
        reasons.append(
            "Full converged uniform run exceeds the declared budget; no depths were truncated"
        )
    return dict(
        depths=grid.tolist(),
        seeds=identities,
        shots_per_circuit=nshots,
        circuit_count=circuits,
        paired_circuit_count=circuits // 2,
        requested_shots=total,
        idle_wait_seconds=idle,
        pulse_execution_seconds=sequence,
        setup_seconds=setup,
        known_runtime_lower_bound_seconds=lower_bound,
        estimated_wall_seconds=wall,
        within_shot_budget=shot_ok,
        within_time_budget=time_ok,
        budget_status=status,
        reasons=reasons,
        depth_zero_role="both empty controls acquired; zero excluded from the decay fit",
    )


def assess_irb_tail(
    depths: Sequence[int] | ArrayLike,
    reference: ArrayLike,
    interleaved: ArrayLike,
    *,
    analysis: Mapping[str, Any],
    tail_fraction: float = 0.01,
    tail_points: int = 3,
    confidence_z: float = 1.96,
) -> dict[str, Any]:
    """
    Check final raw tails and final-fit uncertainty without refitting the analysis.

    Parameters
    ----------
    depths : Sequence[int]
        Actual full uniform grid, including the excluded depth-zero controls.
    reference, interleaved : ArrayLike
        Raw all-shot survival matrices in matched (seed, depth) order.
    analysis : Mapping[str, Any]
        Unmodified final `analyze_irb` summary, including both fit records and
        `quote_as_irb_estimate`. A failed final fit cannot pass this assessment.
    tail_fraction : float, optional
        Maximum remaining initial contrast at the largest depth, default 0.01.
    tail_points : int, optional
        Last positive depths used for the raw floor comparison, default three.
    confidence_z : float, optional
        Local normal confidence multiplier, default 1.96.

    Returns
    -------
    dict[str, Any]
        Passed flag, reasons, raw tail means/seed SEMs and final p upper-tail bounds.
        Missing or unqualified fits retain raw tail statistics without evaluating
        invalid model parameters; malformed raw measurements remain errors.

    Notes
    -----
    Raw tail-floor compatibility is a noncontradiction check, not an equivalence
    proof. It allows amplitude*tail_fraction plus z*(tail_SEM+floor_SE), a
    conservative bound with unknown floor/tail covariance. SEM uses per-seed
    tail averages, preserving within-seed depth correlation. Confidence bounds
    are local fit/statistical diagnostics, not drift or gate-dependent-noise bounds.
    """
    grid = _uniform_depths(depths)
    count = _integer(tail_points, "tail_points")
    fraction = _finite(tail_fraction, "tail_fraction")
    z = _finite(confidence_z, "confidence_z")
    if count > len(grid) - 1 or not 0 < fraction < 1 or z <= 0:
        raise ValueError("Invalid positive tail range or confidence settings")
    arrays = [np.asarray(data, dtype=float) for data in (reference, interleaved)]
    if (
        arrays[0].ndim != 2
        or arrays[0].shape[0] < 4
        or arrays[0].shape[1] != len(grid)
        or arrays[1].shape != arrays[0].shape
    ):
        raise ValueError(
            "Need paired raw (seed,depth) matrices with at least four seeds"
        )
    if any(
        not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1)
        for data in arrays
    ):
        raise ValueError(
            "Raw tail probabilities, including zero controls, must lie in [0,1]"
        )
    reasons = []
    if not analysis.get("quote_as_irb_estimate", False):
        reasons.append("Final IRB analysis is unqualified")
    diagnostics = {}
    for mode, data in zip(("reference", "interleaved"), arrays, strict=True):
        per_seed = data[:, -count:].mean(axis=1)
        measured = float(per_seed.mean())
        sem = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
        diagnostics[mode] = dict(
            model_tail_evaluated=False,
            remaining_contrast_fraction=None,
            upper_remaining_contrast_fraction=None,
            tail_depths=grid[-count:].tolist(),
            raw_tail_mean=measured,
            raw_tail_seed_sem=sem,
            tail_floor_compatible=None,
        )
        fit = analysis.get("fits", {}).get(mode)
        if fit is None:
            reasons.append(f"{mode}: final fit missing")
            continue
        if not fit.get("fit_quality_pass", False):
            reasons.append(f"{mode}: final decay fit unqualified")
            continue
        p = _probability(fit["p"])
        p_error = _finite(fit["errors"]["p"], "p uncertainty")
        floor_error = _finite(fit["errors"]["floor"], "floor uncertainty")
        amplitude = _finite(fit["amplitude"], "amplitude")
        fitted_floor = _finite(fit["floor"], "floor")
        if p_error < 0 or floor_error < 0 or amplitude < 0:
            raise ValueError(
                "Final fit uncertainties and amplitude must be nonnegative"
            )
        upper_p = p + z * p_error
        upper_tail = float(upper_p ** int(grid[-1])) if upper_p < 1 else None
        tail = float(p ** int(grid[-1]))
        if upper_tail is None or upper_tail > fraction:
            reasons.append(
                f"{mode}: final tail is not bounded below the requested contrast fraction"
            )
        tolerance = amplitude * fraction + z * (sem + floor_error)
        compatible = abs(measured - fitted_floor) <= tolerance
        if not compatible:
            reasons.append(f"{mode}: measured tail disagrees with the fitted floor")
        diagnostics[mode] = dict(
            model_tail_evaluated=True,
            remaining_contrast_fraction=tail,
            upper_remaining_contrast_fraction=upper_tail,
            upper_decay_p=upper_p,
            tail_depths=grid[-count:].tolist(),
            raw_tail_mean=measured,
            raw_tail_seed_sem=sem,
            fitted_floor=fitted_floor,
            floor_standard_error=floor_error,
            floor_compatibility_tolerance=tolerance,
            tail_floor_compatible=compatible,
        )
    return dict(
        passed=not reasons,
        reasons=reasons,
        by_mode=diagnostics,
        tail_fraction=fraction,
        confidence_z=z,
        analysis_refitted=False,
        scope="final fit tail plus raw-floor noncontradiction; not independent fidelity or drift proof",
    )
