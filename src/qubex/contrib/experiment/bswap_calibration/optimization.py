"""
Count-driven siZZle nulling and bounded SQUAD optimization for manual runs.

The caller owns the connected CampaignMeasurements session. Nothing here
connects, retunes device settings, changes readout, or writes shared calibration.
Every candidate remains a run-local recipe. Return values are (recipe, summary)
and the caller explicitly accepts a returned recipe. Calibration/validation
shots are separated, and a failed qualification never returns a passing record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from scipy.optimize import minimize

from .local_fit import fit_local_map
from .measurements import save_json, zero_phase_recipe
from .pulses import ideal_circuit_unitary, make_squad_pulse
from .tomography import (
    BASES,
    PAULI,
    SQRT_BASES,
    density_from_counts,
    fit_local_phases,
    sqrt_score,
    state_vector,
)


class QualificationError(RuntimeError):
    """A measured candidate failed its predeclared scientific acceptance gate."""


def _reject(message: str) -> NoReturn:
    raise QualificationError(message)


class ShotBudget:
    """Bound requested hardware shots, including calibration and validation."""

    def __init__(self, maximum: int) -> None:
        """Initialize a maximum requested-shot budget."""
        if isinstance(maximum, bool) or int(maximum) != maximum or maximum <= 0:
            raise ValueError("max_total_shots must be a positive integer")
        self.maximum, self.requested = int(maximum), 0

    def reserve(self, shots: int) -> None:
        """Reserve shots before a hardware request or fail without spending."""
        if isinstance(shots, bool) or int(shots) != shots or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if self.requested + shots > self.maximum:
            _reject("Declared shot budget exhausted before another acquisition")
        self.requested += int(shots)

    def report(self) -> dict[str, int]:
        """Return requested and maximum shot totals."""
        return {"requested_shots": self.requested, "maximum_shots": self.maximum}


def _wrap(x: Any) -> Any:
    return np.angle(np.exp(1j * np.asarray(x)))


def _gate(kind: str) -> str:
    if kind not in ("bswap", "sqrt_bswap"):
        raise ValueError("kind must be bswap or sqrt_bswap")
    return "BSWAP" if kind == "bswap" else "RAW_SQRT_BSWAP"


def _selected(
    measurements: Any, kind: str, recipe: Any, uncorrected: bool = False
) -> dict[str, Any]:
    selected = deepcopy(measurements.recipes)
    selected[kind] = zero_phase_recipe(recipe) if uncorrected else deepcopy(recipe)
    return selected


def _acquire(
    measurements: Any,
    gates: Sequence[Any],
    directory: str | Path,
    label: str,
    *,
    budget: ShotBudget,
    shots: int,
    **kwargs,
) -> dict[str, Any]:
    budget.reserve(shots)
    row = measurements.acquire(gates, directory, label, shots=shots, **kwargs)
    counts = np.asarray(row["counts"])
    if counts.shape != (4,) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Measurement returned malformed four-outcome counts")
    if np.any(counts != np.rint(counts)) or counts.sum() != shots:
        raise ValueError("Measurement counts do not match the requested all-shot total")
    return row


def _hash_recipe(recipe: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            recipe, sort_keys=True, default=lambda v: np.asarray(v).tolist()
        ).encode()
    ).hexdigest()


# Output conditional-coherence axis, input fixed bit, output spectator projector,
# and the phase of the ideal transferred +i/-i matrix element.
_PHASE_CASES = (
    ("active_forward", 0, "0", 1, np.pi / 2),
    ("active_reverse", 0, "1", 0, -np.pi / 2),
    ("passive_forward", 1, "0", 1, np.pi / 2),
    ("passive_reverse", 1, "1", 0, -np.pi / 2),
)
_PHASE_STATES = ("+", "-", "+i", "-i")


def _conditional_coherence(x: Any, y: Any, axis: float, spectator: int) -> complex:
    i, j = (
        (spectator, 2 + spectator) if axis == 0 else (2 * spectator, 2 * spectator + 1)
    )
    return ((x[i] - x[j]) + 1j * (y[i] - y[j])) / 2


def estimate_phase_cycle(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    bootstrap: int = 400,
    seed: int = 9137,
) -> dict[str, Any]:
    """
    Estimate signed integrated ZZ from phase-cycled all-shot counts.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Exactly 32 unique case/phase_state/axis settings with integer counts.
    kind : str
        bswap or sqrt_bswap; ideal transfer visibility is 1 or 1/sqrt(2).
    bootstrap : int, optional
        At least two multinomial bootstrap replicates, default 400.
    seed : int, optional
        Fixed bootstrap seed.

    Returns
    -------
    dict
        Active/passive Phi_ZZ, confidence intervals, visibility, and agreement.
        Phi_ZZ is in radians modulo pi/2; the primitive zeta is 2*Phi_ZZ.

    Raises
    ------
    ValueError
        Missing settings, invalid counts, or invalid bootstrap configuration.

    Notes
    -----
    Spectator-projected coherences are not divided by survival. Intervals
    cover conditional multinomial shot noise, not SPAM or drift systematics.
    """
    _gate(kind)
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 2:
        raise ValueError("bootstrap must be an integer >=2")
    indexed = {
        (r["case"], r["phase_state"], r["axis"]): np.asarray(r["counts"], float)
        for r in rows
    }
    expected = {
        (case[0], state, axis)
        for case in _PHASE_CASES
        for state in _PHASE_STATES
        for axis in "XY"
    }
    if len(rows) != 32 or set(indexed) != expected:
        raise ValueError("Phase cycle requires each of its 32 settings exactly once")
    keys = sorted(indexed)
    counts = np.asarray([indexed[key] for key in keys])
    if (
        counts.shape != (32, 4)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts.sum(1) <= 0)
    ):
        raise ValueError("Invalid phase-cycle counts")
    if np.any(counts != np.rint(counts)):
        raise ValueError("Phase-cycle input must be integer counts")
    positions = {key: i for i, key in enumerate(keys)}

    def estimate(probabilities: Any) -> Any:
        phases, amplitudes = {}, {}
        for name, axis, _, spectator, ideal in _PHASE_CASES:
            c = {}
            for state in _PHASE_STATES:
                x, y = (probabilities[positions[(name, state, b)]] for b in "XY")
                c[state] = _conditional_coherence(x, y, axis, spectator)
            z = (c["+"] - c["-"] + 1j * (c["+i"] - c["-i"])) / 2
            phases[name] = float(_wrap(np.angle(z) - ideal))
            amplitudes[name] = float(abs(z))
        a = float(_wrap(phases["active_reverse"] - phases["active_forward"]) / 4)
        p = float(_wrap(phases["passive_reverse"] - phases["passive_forward"]) / 4)
        return np.array([a, p, (a + p) / 2]), amplitudes, phases

    probabilities = counts / counts.sum(1)[:, None]
    nominal, amplitudes, phases = estimate(probabilities)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(bootstrap)):
        sampled = np.asarray(
            [
                rng.multinomial(int(n), p) / n
                for n, p in zip(counts.sum(1), probabilities, strict=True)
            ]
        )
        draws.append(estimate(sampled)[0])
    draws = nominal + _wrap(4 * (np.asarray(draws) - nominal)) / 4
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    ideal_visibility = 1.0 if kind == "bswap" else 1 / np.sqrt(2)
    return {
        "Phi_ZZ_active_rad": float(nominal[0]),
        "Phi_ZZ_passive_rad": float(nominal[1]),
        "Phi_ZZ_mean_rad": float(nominal[2]),
        "zz_phase_rad": float(2 * nominal[2]),
        "ci95_active_rad": [float(low[0]), float(high[0])],
        "ci95_passive_rad": [float(low[1]), float(high[1])],
        "ci95_mean_rad": [float(low[2]), float(high[2])],
        "direction_disagreement_rad": float(abs(nominal[0] - nominal[1])),
        "coherence_amplitudes": amplitudes,
        "phases_after_ideal_removal": phases,
        "minimum_coherence_fraction": min(amplitudes.values()) / ideal_visibility,
        "ideal_transfer_coherence": ideal_visibility,
        "convention": "U_ZZ=exp(-i Phi_ZZ ZZ); zz_phase_rad=2*Phi_ZZ",
        "uncertainty": "conditional multinomial shots only; no SPAM or drift systematic coverage",
    }


def phase_cycle_zz(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int = 512,
    bootstrap: int = 400,
    seed: int = 7001,
    budget: ShotBudget | None = None,
) -> dict[str, Any]:
    """
    Acquire a randomized 32-setting ZZ phase cycle on one exact waveform.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing connected count port; this function never connects.
    kind : str
        bswap or sqrt_bswap.
    recipe : Mapping[str, Any]
        Exact waveform to probe with explicit zero local calibration phases.
    directory : str or Path
        Fresh output directory for partial and final count records.
    shots : int, optional
        Shots per setting, default 512.
    bootstrap : int, optional
        Multinomial confidence replicates, default 400.
    seed : int, optional
        Acquisition-order seed.
    budget : ShotBudget or None, optional
        Shared upper bound on requested shots.

    Returns
    -------
    dict
        Recipe, session, raw count rows, and signed-ZZ estimate.

    Notes
    -----
    Performs hardware acquisition through the caller's port and writes
    incremental evidence. No device configuration or shared calibration is
    changed; every input has the same calibrated gate start.
    """
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 2:
        raise ValueError("bootstrap must be an integer >=2")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    budget = budget or ShotBudget(32 * shots)
    selected = _selected(measurements, kind, recipe, uncorrected=True)
    specs = [
        (case, state, axis)
        for case in _PHASE_CASES
        for state in _PHASE_STATES
        for axis in "XY"
    ]
    rows = []
    for index in np.random.default_rng(seed).permutation(len(specs)):
        case, state, axis = specs[index]
        name, measured_axis, fixed, _, _ = case
        prepared = (fixed, state) if measured_axis == 0 else (state, fixed)
        basis = axis + "Z" if measured_axis == 0 else "Z" + axis
        row = _acquire(
            measurements,
            [_gate(kind)],
            directory,
            f"zz_{name}_{state}_{axis}",
            prepared=prepared,
            basis=basis,
            recipes=selected,
            budget=budget,
            shots=shots,
        )
        rows.append(
            {
                "case": name,
                "phase_state": state,
                "axis": axis,
                "counts": row["counts"],
                "measurement": row,
            }
        )
        save_json(
            directory / "phase_cycle_partial.json",
            {"rows": rows, "budget": budget.report()},
        )
    estimate = estimate_phase_cycle(rows, kind=kind, bootstrap=bootstrap, seed=seed + 1)
    result = {
        "kind": kind,
        "recipe": deepcopy(recipe),
        "recipe_hash": _hash_recipe(recipe),
        "estimate": estimate,
        "rows": rows,
        "shots_per_setting": int(shots),
        "session_id": measurements.session_id,
        "budget": budget.report(),
    }
    save_json(directory / "phase_cycle.json", result)
    return result


def _population_check(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int,
    budget: ShotBudget,
) -> dict[str, Any]:
    selected = _selected(measurements, kind, recipe)
    u = ideal_circuit_unitary([_gate(kind)])
    rows = []
    for state in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")):
        row = _acquire(
            measurements,
            [_gate(kind)],
            directory,
            "population_" + "".join(state),
            prepared=state,
            recipes=selected,
            budget=budget,
            shots=shots,
        )
        p = np.asarray(row["counts"]) / shots
        ideal = np.abs(u @ state_vector(state)) ** 2
        rows.append(
            {
                "state": state,
                "raw_probabilities": p.tolist(),
                "target_probabilities": ideal.tolist(),
                "population_agreement": float(1 - 0.5 * np.sum(abs(p - ideal))),
                "measurement": row,
            }
        )
    result = {
        "minimum_population_agreement": min(r["population_agreement"] for r in rows),
        "rows": rows,
        "claim": "raw classified distribution agreement, not gate fidelity or leakage resolved",
    }
    save_json(Path(directory) / "population_check.json", result)
    return result


def recenter_amplitude_frequency(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int = 256,
    amplitude_span: float = 0.012,
    frequency_span_mhz: float = 0.6,
    max_amplitude: float = 0.99,
    grid_points: int = 4,
    budget: ShotBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fit a bounded local map at fixed complete duration and ramp.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing connected count port.
    kind : str
        bswap or sqrt_bswap.
    recipe : Mapping[str, Any]
        Starting recipe, copied without mutation.
    directory : str or Path
        Output directory for raw-count references, fit, and confirmation.
    shots : int, optional
        Shots per setting, default 256.
    amplitude_span : float, optional
        Symmetric command-amplitude span, default 0.012.
    frequency_span_mhz : float, optional
        Symmetric carrier span in MHz, default 0.6.
    max_amplitude : float, optional
        Maximum main command amplitude, default 0.99.
    grid_points : int, optional
        At least four points on each axis.
    budget : ShotBudget or None, optional
        Shared requested-shot budget.

    Returns
    -------
    tuple[dict, dict]
        Recentered recipe and fit/population evidence.

    Raises
    ------
    QualificationError
        A local quadratic is unsupported by its shot residuals.

    Notes
    -----
    Performs acquisitions and writes files. Root scores include even
    coherence; full scores use both transfers. Unresolved map modulation
    retains the seed instead of choosing an arbitrary noise optimum.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if grid_points < 4 or amplitude_span <= 0 or frequency_span_mhz <= 0:
        raise ValueError("Need >=4 points and positive amp/frequency spans")
    bases = ("ZZ",) if kind == "bswap" else SQRT_BASES
    budget = budget or ShotBudget((2 * grid_points**2 * len(bases) + 4) * shots)
    aa = np.linspace(
        max(0.01, recipe["amplitude"] - amplitude_span),
        min(max_amplitude, recipe["amplitude"] + amplitude_span),
        grid_points,
    )
    ff = (
        recipe["frequency_ghz"]
        + np.linspace(-frequency_span_mhz, frequency_span_mhz, grid_points) / 1000
    )
    score = np.full((2, grid_points, grid_points), np.nan)
    variance = np.full_like(score, np.nan)
    requests = [
        (d, a, f)
        for d in range(2)
        for a in range(grid_points)
        for f in range(grid_points)
    ]
    for index in np.random.default_rng(891).permutation(len(requests)):
        d, a, f = requests[index]
        candidate = {
            **deepcopy(recipe),
            "amplitude": float(aa[a]),
            "frequency_ghz": float(ff[f]),
        }
        selected = _selected(measurements, kind, candidate, uncorrected=True)
        counts = []
        for basis in bases:
            row = _acquire(
                measurements,
                [_gate(kind)],
                directory,
                f"map_{d}_{a}_{f}_{basis}",
                prepared=(("0", "0"), ("1", "1"))[d],
                basis=basis,
                recipes=selected,
                budget=budget,
                shots=shots,
            )
            counts.append(row["counts"])
        if kind == "bswap":
            p = counts[0][3 if d == 0 else 0] / shots
            score[d, a, f], variance[d, a, f] = (
                p,
                max(p * (1 - p) / shots, 1 / (4 * shots**2)),
            )
        else:
            score[d, a, f], variance[d, a, f] = sqrt_score(counts)
        np.savez_compressed(
            directory / "local_map_partial.npz",
            amplitudes=aa,
            frequencies_ghz=ff,
            scores=score,
            variances=variance,
        )
    try:
        fit = fit_local_map(aa, ff, score, shots, score_variance=variance)
    except ValueError as error:
        raise QualificationError(
            f"Local amplitude/frequency fit failed: {error}"
        ) from error
    save_json(directory / "local_fit.json", fit)
    if fit["reduced_chi2"] > 5 or not np.isfinite(fit["ranking_score"]):
        _reject("Local quadratic map is not supported by its shot residuals")
    resolved = np.ptp(fit["prediction"]) > 2 * np.sqrt(np.median(variance))
    record = deepcopy(recipe)
    if resolved:
        record.update(amplitude=fit["amplitude"], frequency_ghz=fit["frequency_ghz"])
    fit["selection"] = (
        "quadratic lower-bound maximum"
        if resolved
        else "retain seed: local modulation unresolved"
    )
    save_json(directory / "local_fit.json", fit)
    population = _population_check(
        measurements,
        kind,
        record,
        directory / "confirmation",
        shots=shots,
        budget=budget,
    )
    record["local_refinement_directory"] = str(directory)
    save_json(directory / "recentered_recipe.json", record)
    return record, {
        "fit": fit,
        "population": population,
        "duration_fixed_ns": recipe["duration_ns"],
        "budget": budget.report(),
    }


def _refresh_phases(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int,
    budget: ShotBudget,
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selected = _selected(measurements, kind, recipe, uncorrected=True)
    states = [("0", "+"), ("1", "+"), ("+", "0"), ("+", "1")]
    rhos, all_counts = [], []
    for index, state in enumerate(states):
        counts = []
        for basis in BASES:
            row = _acquire(
                measurements,
                [_gate(kind)],
                directory,
                f"phase_{index}_{basis}",
                prepared=state,
                basis=basis,
                recipes=selected,
                budget=budget,
                shots=shots,
            )
            counts.append(row["counts"])
        rhos.append(density_from_counts(counts))
        all_counts.append(counts)
    np.savez_compressed(
        directory / "phase_counts.npz", counts=all_counts, states=states, bases=BASES
    )
    try:
        fit = fit_local_phases(kind, states, rhos)
    except ValueError as error:
        raise QualificationError(
            f"Measured phase calibration failed: {error}"
        ) from error
    if fit["coherence_residual_rms"] > 0.08:
        _reject("Local-phase model does not describe measured coherences")
    record = deepcopy(recipe)
    record.update(
        phase_calibration=fit,
        pre_vz_rad=fit["pre_vz_rad"],
        post_vz_rad=fit["post_vz_rad"],
        zz_phase_rad=fit["zz_phase_rad"],
        phase_data_directory=str(directory),
        phase_status="measured; independent validation pending",
    )
    save_json(directory / "phase_calibration.json", record)
    return record


def _interval_sign(estimate: Mapping[str, Any]) -> int:
    intervals = [estimate[k] for k in ("ci95_active_rad", "ci95_passive_rad")]
    if all(low > 0 for low, high in intervals):
        return 1
    if all(high < 0 for low, high in intervals):
        return -1
    return 0


def _null_pass(
    estimate: Mapping[str, Any],
    tolerance: float,
    disagreement: float,
    coherence_fraction: float,
) -> bool:
    return bool(
        all(
            -tolerance < estimate[k][0] <= estimate[k][1] < tolerance
            for k in ("ci95_active_rad", "ci95_passive_rad")
        )
        and estimate["direction_disagreement_rad"] < disagreement
        and estimate["minimum_coherence_fraction"] >= coherence_fraction
    )


def calibrate_sizzle(
    measurements: Any,
    kind: str,
    directory: str | Path,
    *,
    recipe: Any = None,
    shots: int = 512,
    validation_shots: int = 8192,
    tolerance_phi_zz_rad: float | None = None,
    ratio_grid: Sequence[float] = (0.0, 0.015, 0.03, 0.045, 0.06),
    probe_ratio: float = 0.03,
    minimum_coherence_fraction: float = 0.6,
    minimum_population_agreement: float = 0.65,
    maximum_direction_disagreement_rad: float = 0.02,
    max_refinements: int = 3,
    max_total_shots: int = 2_000_000,
    bootstrap: int = 400,
    recenter: bool = True,
    budget: ShotBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Qualify a same-session exact-waveform integrated-ZZ null.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing live count port; connection ownership remains with the caller.
    kind : str
        bswap or sqrt_bswap, calibrated independently.
    directory : str or Path
        Fresh attempt directory; existing protocol evidence is never overwritten.
    recipe : Mapping[str, Any] or None, optional
        Starting waveform, or the port's current recipe.
    shots : int, optional
        Training shots per setting, default 512.
    validation_shots : int, optional
        Fresh validation shots per setting, default 8192.
    tolerance_phi_zz_rad : float or None, optional
        Entire-CI equivalence tolerance; defaults to 0.03 full or 0.02 root.
    ratio_grid : Sequence[float], optional
        Increasing same-carrier amplitude ratios from zero, bounded by 0.1.
    probe_ratio : float, optional
        Amplitude ratio for the four-phase scout, default 0.03.
    minimum_coherence_fraction : float, optional
        Minimum visibility relative to the ideal full/root transfer, default 0.6.
    minimum_population_agreement : float, optional
        Minimum four-input classified distribution agreement, default 0.65.
    maximum_direction_disagreement_rad : float, optional
        Active/passive Phi_ZZ agreement threshold, default 0.02 radians.
    max_refinements : int, optional
        Maximum training-only bracket refinements, default three.
    max_total_shots : int, optional
        Requested-shot ceiling, default two million; not a spending target.
    bootstrap : int, optional
        Shot-bootstrap replicates, default 400.
    recenter : bool, optional
        Refit amplitude/carrier at fixed duration with the tone on.
    budget : ShotBudget or None, optional
        Shared parent budget, used instead of creating a separate budget.

    Returns
    -------
    tuple[dict, dict]
        Independently qualified recipe and complete acceptance summary.

    Raises
    ------
    QualificationError
        Unresolved phase response, missing sign bracket, failed transfer,
        exhausted budget, or failed independent null validation.
    FileExistsError
        The attempt directory already contains a protocol.

    Notes
    -----
    Acquires and incrementally saves hardware counts through the existing
    port. No port recipe, device setting, readout, or shared calibration is
    mutated. The ratio grid stops once a valid sign bracket is measured.
    Duration and ramp remain fixed. A failed independent validation stops,
    rather than tuning on those shots and calling them held out again.
    Qualification is pulse-integrated ZZ, not pointwise ZZ or gate fidelity.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _gate(kind)
    if (directory / "protocol.json").exists():
        raise FileExistsError(
            "Use a fresh siZZle attempt directory; prior evidence is retained"
        )
    base = deepcopy(measurements.recipes[kind] if recipe is None else recipe)
    tolerance = (
        (0.03 if kind == "bswap" else 0.02)
        if tolerance_phi_zz_rad is None
        else float(tolerance_phi_zz_rad)
    )
    ratios = np.asarray(ratio_grid, float)
    if (
        not 0 < tolerance < np.pi / 8
        or not np.isfinite(ratios).all()
        or len(ratios) < 2
    ):
        raise ValueError("Invalid predeclared ZZ tolerance or ratio grid")
    if (
        ratios[0] != 0
        or np.any(np.diff(ratios) <= 0)
        or ratios[-1] > 0.1
        or not 0 < probe_ratio <= ratios[-1]
    ):
        raise ValueError("Ratio grid must increase from zero and remain <=0.1")
    if (
        isinstance(max_refinements, bool)
        or int(max_refinements) != max_refinements
        or max_refinements < 1
    ):
        raise ValueError("max_refinements must be a positive integer")
    if not (
        0 < minimum_coherence_fraction <= 1
        and 0 < minimum_population_agreement <= 1
        and 0 < maximum_direction_disagreement_rad < np.pi / 4
    ):
        raise ValueError("Invalid predeclared exchange/coherence/direction criteria")
    budget = budget or ShotBudget(max_total_shots)
    summary = {
        "kind": kind,
        "qualified": False,
        "status": "calibrating",
        "tolerance_phi_zz_rad": tolerance,
        "ratio_grid": ratios.tolist(),
        "probe_ratio": probe_ratio,
        "minimum_coherence_fraction": minimum_coherence_fraction,
        "minimum_population_agreement": minimum_population_agreement,
        "maximum_direction_disagreement_rad": maximum_direction_disagreement_rad,
        "shots_per_setting": shots,
        "validation_shots_per_setting": validation_shots,
        "session_id": measurements.session_id,
        "fixed_duration_ns": base["duration_ns"],
        "fixed_ramp_ns": base.get("ramp_ns", 16.0),
    }
    save_json(directory / "protocol.json", summary)
    try:
        off = {**deepcopy(base), "cancel_amplitude_ratio": 0.0, "cancel_phase_rad": 0.0}
        off_cycle = phase_cycle_zz(
            measurements,
            kind,
            off,
            directory / "same_session_off",
            shots=shots,
            bootstrap=bootstrap,
            seed=11001,
            budget=budget,
        )
        phase_rows = []
        for index, phase in enumerate(np.arange(4) * np.pi / 2):
            probe = {
                **deepcopy(base),
                "cancel_amplitude_ratio": probe_ratio,
                "cancel_phase_rad": float(phase),
            }
            result = phase_cycle_zz(
                measurements,
                kind,
                probe,
                directory / f"phase_{index}",
                shots=shots,
                bootstrap=bootstrap,
                seed=11010 + index,
                budget=budget,
            )
            phase_rows.append(result)
        estimates = [r["estimate"] for r in phase_rows]
        if (
            min(e["minimum_coherence_fraction"] for e in estimates)
            < minimum_coherence_fraction
        ):
            _reject(
                "Relative-phase scout lost transferred coherence; recenter/review before retry"
            )
        x = np.arange(4) * np.pi / 2
        y = np.array([e["Phi_ZZ_mean_rad"] for e in estimates])
        off_phi = off_cycle["estimate"]["Phi_ZZ_mean_rad"]
        y = off_phi + _wrap(4 * (y - off_phi)) / 4
        se = np.array(
            [(e["ci95_mean_rad"][1] - e["ci95_mean_rad"][0]) / 3.92 for e in estimates]
        )
        design = np.column_stack([np.ones(4), np.cos(x), np.sin(x)])
        weights = 1 / np.maximum(se, 0.001)
        coefficient = np.linalg.lstsq(
            design * weights[:, None], y * weights, rcond=None
        )[0]
        if np.hypot(*coefficient[1:]) < 3 * np.mean(np.maximum(se, 0.001)):
            _reject("Relative siZZle phase response is unresolved")
        phase = float(
            _wrap(
                np.arctan2(coefficient[2], coefficient[1])
                + (np.pi if off_phi >= 0 else 0)
            )
        )
        summary.update(
            selected_phase_rad=phase,
            harmonic_coefficients=coefficient.tolist(),
            phase_reference_id=f"{summary['session_id']}:{_hash_recipe(base)[:16]}:{directory.name}",
            phase_map=[
                {"phase": float(x[i]), "estimate": estimates[i]} for i in range(4)
            ],
        )
        save_json(directory / "phase_reference.json", summary)
        bracket_rows = []
        for index, ratio in enumerate(ratios):
            candidate = {
                **deepcopy(base),
                "cancel_amplitude_ratio": float(ratio),
                "cancel_phase_rad": phase,
            }
            if recenter and ratio > 0:
                candidate, refinement = recenter_amplitude_frequency(
                    measurements,
                    kind,
                    candidate,
                    directory / f"bracket_{index}" / "recenter",
                    shots=shots,
                    budget=budget,
                )
                population = refinement["population"]
            else:
                population = _population_check(
                    measurements,
                    kind,
                    candidate,
                    directory / f"bracket_{index}" / "population",
                    shots=shots,
                    budget=budget,
                )
            result = phase_cycle_zz(
                measurements,
                kind,
                candidate,
                directory / f"bracket_{index}" / "zz",
                shots=shots,
                bootstrap=bootstrap,
                seed=11100 + index,
                budget=budget,
            )
            valid = (
                result["estimate"]["minimum_coherence_fraction"]
                >= minimum_coherence_fraction
                and population["minimum_population_agreement"]
                >= minimum_population_agreement
                and result["estimate"]["direction_disagreement_rad"]
                < maximum_direction_disagreement_rad
            )
            bracket_rows.append(
                {
                    "ratio": float(ratio),
                    "recipe": candidate,
                    "estimate": result["estimate"],
                    "population": population,
                    "valid_exchange": bool(valid),
                }
            )
            save_json(directory / "sign_bracket.json", bracket_rows)
            signed = [
                r
                for r in bracket_rows
                if r["valid_exchange"] and _interval_sign(r["estimate"]) != 0
            ]
            if any(
                _interval_sign(a["estimate"]) * _interval_sign(b["estimate"]) == -1
                and abs(
                    a["estimate"]["Phi_ZZ_mean_rad"] - b["estimate"]["Phi_ZZ_mean_rad"]
                )
                < np.pi / 4
                for a, b in pairwise(signed)
            ):
                break  # The declared grid is an upper budget, not mandatory spending.
        # A grid point whose CI contains zero must not hide signed endpoints
        # on either side. It is a candidate, not itself a sign-bracket endpoint.
        valid = [
            r
            for r in bracket_rows
            if r["valid_exchange"] and _interval_sign(r["estimate"]) != 0
        ]
        brackets = [
            (a, b)
            for a, b in pairwise(valid)
            if _interval_sign(a["estimate"]) * _interval_sign(b["estimate"]) == -1
            and abs(a["estimate"]["Phi_ZZ_mean_rad"] - b["estimate"]["Phi_ZZ_mean_rad"])
            < np.pi / 4
        ]
        if not brackets:
            _reject("No measured same-branch ZZ sign bracket with preserved exchange")
        left, right = min(
            brackets, key=lambda pair: pair[1]["ratio"] - pair[0]["ratio"]
        )
        summary["bracket"] = [
            {"ratio": r["ratio"], "estimate": r["estimate"]} for r in (left, right)
        ]
        calibrated = None
        for attempt in range(int(max_refinements)):
            y0, y1 = (r["estimate"]["Phi_ZZ_mean_rad"] for r in (left, right))
            ratio = left["ratio"] - y0 * (right["ratio"] - left["ratio"]) / (y1 - y0)
            if not left["ratio"] < ratio < right["ratio"]:
                _reject("ZZ interpolation left the measured bracket")
            candidate = {
                **deepcopy(base),
                "cancel_amplitude_ratio": float(ratio),
                "cancel_phase_rad": phase,
            }
            attempt_dir = directory / f"null_calibration_{attempt}"
            if recenter:
                candidate, _ = recenter_amplitude_frequency(
                    measurements,
                    kind,
                    candidate,
                    attempt_dir / "recenter",
                    shots=shots,
                    budget=budget,
                )
            candidate = _refresh_phases(
                measurements,
                kind,
                candidate,
                attempt_dir / "vz",
                shots=shots,
                budget=budget,
            )
            cycle = phase_cycle_zz(
                measurements,
                kind,
                candidate,
                attempt_dir / "training_zz",
                shots=shots,
                bootstrap=bootstrap,
                seed=11200 + attempt,
                budget=budget,
            )
            estimate = cycle["estimate"]
            if abs(estimate["Phi_ZZ_mean_rad"]) < tolerance / 2:
                calibrated = candidate
                break
            row = {
                "ratio": float(ratio),
                "recipe": candidate,
                "estimate": estimate,
                "valid_exchange": True,
            }
            if _interval_sign(estimate) == _interval_sign(left["estimate"]):
                left = row
            elif _interval_sign(estimate) == _interval_sign(right["estimate"]):
                right = row
            else:
                _reject(
                    "Calibration ZZ uncertainty is too broad to refine its sign bracket"
                )
        if calibrated is None:
            _reject(
                "ZZ calibration did not reach the validation neighborhood within budget"
            )
        calibrated.update(
            phase_reference_id=summary["phase_reference_id"],
            phase_reference_session_id=summary["session_id"],
            tolerance_phi_zz_rad=tolerance,
            null_shot_interval_passed=False,
            frozen_at=datetime.now().astimezone().isoformat(),
        )
        save_json(directory / "frozen_recipe.json", calibrated)
        validation = phase_cycle_zz(
            measurements,
            kind,
            calibrated,
            directory / "independent_zz",
            shots=validation_shots,
            bootstrap=bootstrap,
            seed=11991,
            budget=budget,
        )
        population = _population_check(
            measurements,
            kind,
            calibrated,
            directory / "independent_population",
            shots=validation_shots,
            budget=budget,
        )
        passed = (
            _null_pass(
                validation["estimate"],
                tolerance,
                maximum_direction_disagreement_rad,
                minimum_coherence_fraction,
            )
            and population["minimum_population_agreement"]
            >= minimum_population_agreement
        )
        summary.update(
            validation=validation["estimate"],
            population_validation=population,
            budget=budget.report(),
            qualified=bool(passed),
            status="qualified_integrated_ZZ_null"
            if passed
            else "independent_validation_failed",
            claim="same-session pulse-integrated ZZ null; not pointwise ZZ=0, leakage resolved, or gate fidelity",
        )
        save_json(directory / "sizzle_summary.json", summary)
        if not passed:
            _reject("Frozen siZZle candidate failed fresh ZZ/population validation")
        calibrated.update(
            null_shot_interval_passed=True,
            null_validation=validation["estimate"],
            sizzle_calibration_directory=str(directory),
            zz_phase_rad=validation["estimate"]["zz_phase_rad"],
            phase_status="measured VZ; independent integrated-ZZ/population validation passed",
        )
        save_json(directory / "qualified_recipe.json", calibrated)
    except Exception as error:
        summary.update(
            qualified=False, status="failed", error=str(error), budget=budget.report()
        )
        save_json(directory / "sizzle_summary.json", summary)
        raise
    return calibrated, summary


def _overlap_statistics(counts: Any, target: Any) -> tuple[float, float]:
    """Linear tomography overlap and independent-setting shot standard error."""
    counts = np.asarray(counts, float)
    probabilities = counts / counts.sum(1)[:, None]
    one = np.array([1.0, 1.0, -1.0, -1.0])
    two = np.array([1.0, -1.0, 1.0, -1.0])
    joint = one * two
    weights = []
    for a, b in BASES:
        ta = np.vdot(target, np.kron(PAULI[a], PAULI["I"]) @ target).real
        tb = np.vdot(target, np.kron(PAULI["I"], PAULI[b]) @ target).real
        tab = np.vdot(target, np.kron(PAULI[a], PAULI[b]) @ target).real
        weights.append((ta * one / 3 + tb * two / 3 + tab * joint) / 4)
    weights = np.asarray(weights)
    means = np.sum(weights * probabilities, axis=1)
    variance = np.sum(
        (np.sum(weights**2 * probabilities, axis=1) - means**2) / counts.sum(1)
    )
    return float(0.25 + means.sum()), float(np.sqrt(max(variance, 0)))


def short_gate_score(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int = 512,
    validation: bool = False,
    budget: ShotBudget | None = None,
) -> dict[str, Any]:
    """
    Measure an independent-input repeated-coherence optimization score.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing live count port.
    kind : str
        bswap or sqrt_bswap.
    recipe : Mapping[str, Any]
        Calibrated candidate copied into requests without mutating the port.
    directory : str or Path
        Partial and final count-evidence directory.
    shots : int, optional
        Shots per physical analysis setting, default 512.
    validation : bool, optional
        Use held-out inputs, repetitions, and delays instead of training cases.
    budget : ShotBudget or None, optional
        Shared requested-shot budget.

    Returns
    -------
    dict
        Raw state-overlap mean, shot error, lower-bound ranking score, and
        four-input population evidence.

    Notes
    -----
    Acquires all nine physical analysis settings. Training uses ++/+i0 at
    repetitions 1/2; validation changes inputs and includes repetitions 3/4
    and placement delays. Scores are neither clipped nor gate fidelities.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = [
        (("+", "+"), 1, 0.0),
        (("+i", "0"), 1, 0.0),
        (("+", "+"), 2, 0.0),
        (("+i", "0"), 2, 0.0),
    ]
    if validation:
        cases = [
            (("+", "+i"), 1, 2.0),
            (("0", "+i"), 2, 64.0),
            (("+i", "+"), 3, 16.0),
            (("+", "+"), 4, 256.0),
        ]
    budget = budget or ShotBudget((9 * len(cases) + 4) * shots)
    selected, rows = _selected(measurements, kind, recipe), []
    for i, (prepared, repeats, delay) in enumerate(cases):
        gates, counts = [_gate(kind)] * repeats, []
        for basis in BASES:
            row = _acquire(
                measurements,
                gates,
                directory,
                f"score_{i}_{basis}",
                recipes=selected,
                prepared=prepared,
                basis=basis,
                delay_ns=delay,
                budget=budget,
                shots=shots,
            )
            counts.append(row["counts"])
        target = ideal_circuit_unitary(gates) @ state_vector(prepared)
        value, error = _overlap_statistics(counts, target)
        rows.append(
            {
                "prepared": prepared,
                "repeats": repeats,
                "delay_ns": delay,
                "counts": counts,
                "raw_state_overlap": value,
                "shot_standard_error": error,
            }
        )
        save_json(directory / "score_partial.json", rows)
    mean = float(np.mean([r["raw_state_overlap"] for r in rows]))
    se = float(np.sqrt(sum(r["shot_standard_error"] ** 2 for r in rows)) / len(rows))
    population = _population_check(
        measurements, kind, recipe, directory / "population", shots=shots, budget=budget
    )
    result = {
        "score": mean,
        "shot_standard_error": se,
        "ranking_score": mean - 1.96 * se,
        "minimum_state_overlap": min(r["raw_state_overlap"] for r in rows),
        "rows": rows,
        "population": population,
        "validation": bool(validation),
        "budget": budget.report(),
        "claim": "raw fixed-ideal state/repetition score; no clipping, postselection, or gate-fidelity interpretation",
    }
    save_json(directory / "gate_score.json", result)
    return result


def optimize_squad(
    measurements: Any,
    kind: str,
    directory: str | Path,
    *,
    max_evaluations: int = 12,
    shots: int = 512,
    validation_shots: int = 2048,
    max_total_shots: int = 4_000_000,
    minimum_validation_score: float = 0.65,
    minimum_population_agreement: float = 0.65,
    recenter: bool = True,
    null_validation_shots: int = 8192,
    bootstrap: int = 400,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Optimize a fixed pulse family using measured coherent gate scores.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Existing live count port with fixed measured Rabi conversion.
    kind : str
        bswap or sqrt_bswap.
    directory : str or Path
        Fresh attempt directory.
    max_evaluations : int, optional
        At least three and at most this many measured Nelder-Mead trials;
        default 12, with early convergence allowed.
    shots : int, optional
        Training shots per setting, default 512.
    validation_shots : int, optional
        Fresh held-out shots per setting, default 2048.
    max_total_shots : int, optional
        Global requested-shot ceiling, default four million.
    minimum_validation_score : float, optional
        Held-out state-score acceptance threshold, default 0.65.
    minimum_population_agreement : float, optional
        Four-input population acceptance threshold, default 0.65.
    recenter : bool, optional
        Alternate bounded amplitude/carrier refinement with shape changes.
    null_validation_shots : int, optional
        Fresh per-setting shots for selected-shape siZZle requalification.
    bootstrap : int, optional
        Multinomial replicates for siZZle uncertainty.

    Returns
    -------
    tuple[dict, dict]
        Frozen independently validated recipe and optimization/comparison record.

    Raises
    ------
    QualificationError
        No valid candidate, exhausted budget, or failed independent checks.
    FileExistsError
        Existing attempt evidence would be overwritten.

    Notes
    -----
    Acquires and saves hardware counts through the existing port. Optimize
    design_delta_scale within [0.6,1.6] and CD strength within [0.3,1.5],
    while measured K, total duration, ramp, and window stay fixed. Every
    candidate refreshes measured VZ before independent training scores.
    The selected siZZle-ON shape separately requalifies its exact-waveform
    ZZ null. Held-out data never choose the optimizer candidate. Neither
    the port's recipes nor shared calibration are mutated. State-score
    improvement is not gate fidelity or a measurement of strong-drive gain.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _gate(kind)
    if (directory / "protocol.json").exists():
        raise FileExistsError(
            "Use a fresh SQUAD attempt directory; prior evidence is retained"
        )
    if (
        isinstance(max_evaluations, bool)
        or int(max_evaluations) != max_evaluations
        or max_evaluations < 3
    ):
        raise ValueError("max_evaluations must be an integer >=3")
    base = deepcopy(measurements.recipes[kind])
    bounds = np.array([[0.6, 1.6], [0.3, 1.5]])
    x0 = np.array([base.get("design_delta_scale", 1.0), base.get("cd_strength", 1.0)])
    if np.any(x0 < bounds[:, 0]) or np.any(x0 > bounds[:, 1]):
        raise ValueError("Starting shape is outside the predeclared optimizer bounds")
    budget = ShotBudget(max_total_shots)
    trials, summary = (
        [],
        {
            "kind": kind,
            "qualified": False,
            "status": "optimizing",
            "max_evaluations": int(max_evaluations),
            "bounds": bounds.tolist(),
            "fixed_duration_ns": base["duration_ns"],
            "fixed_ramp_ns": base.get("ramp_ns", 16.0),
            "fixed_window": base.get("window", {"type": "hann"}),
            "rabi_conversion_fixed": float(measurements.rabi_scale),
            "session_id": measurements.session_id,
        },
    )
    save_json(directory / "protocol.json", summary)

    def objective(parameters: Any) -> float:
        if len(trials) >= max_evaluations:
            _reject("Optimizer attempted to exceed its evaluation budget")
        index, trial_dir = len(trials), directory / f"trial_{len(trials):02d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        candidate = {
            **deepcopy(base),
            "design_delta_scale": float(parameters[0]),
            "cd_strength": float(parameters[1]),
        }
        row = {
            "index": index,
            "parameters": np.asarray(parameters).tolist(),
            "status": "started",
        }
        try:
            # Reject headroom/invalid design before requesting hardware shots.
            try:
                make_squad_pulse(
                    candidate,
                    rabi_ghz_per_amplitude=measurements.rabi_scale,
                    transition_frequency_ghz=measurements.references[
                        measurements.qubits[0]
                    ],
                )
            except ValueError as error:
                raise QualificationError(
                    f"Invalid candidate waveform: {error}"
                ) from error
            if recenter:
                candidate, _ = recenter_amplitude_frequency(
                    measurements,
                    kind,
                    candidate,
                    trial_dir / "recenter",
                    shots=shots,
                    budget=budget,
                )
            candidate = _refresh_phases(
                measurements,
                kind,
                candidate,
                trial_dir / "vz",
                shots=shots,
                budget=budget,
            )
            score = short_gate_score(
                measurements,
                kind,
                candidate,
                trial_dir / "training",
                shots=shots,
                budget=budget,
            )
            if (
                score["population"]["minimum_population_agreement"]
                < minimum_population_agreement
            ):
                _reject("Candidate lost the intended population mapping")
            row.update(status="scored", recipe=candidate, score=score)
            value = -score["ranking_score"]
        except QualificationError as error:
            if "budget" in str(error).lower():
                raise
            row.update(status="rejected", error=str(error))
            value = 1000.0
        trials.append(row)
        save_json(
            directory / "optimization_trials.json",
            {"trials": trials, "budget": budget.report()},
        )
        return value

    try:
        simplex = np.array(
            [
                x0,
                x0 + np.array([0.15 if x0[0] <= 1.45 else -0.15, 0.0]),
                x0 + np.array([0.0, 0.15 if x0[1] <= 1.35 else -0.15]),
            ]
        )
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxfev": int(max_evaluations),
                "initial_simplex": simplex,
                "xatol": 0.025,
                "fatol": 0.002,
                "adaptive": False,
            },
        )
        scored = [row for row in trials if row["status"] == "scored"]
        if not scored:
            _reject("No SQUAD candidate retained a measurable coherent gate")
        winner = max(scored, key=lambda row: row["score"]["ranking_score"])
        chosen = deepcopy(winner["recipe"])
        save_json(directory / "training_selected_recipe.json", chosen)
        if chosen.get("cancel_amplitude_ratio", 0.0) > 0:
            chosen, null_summary = calibrate_sizzle(
                measurements,
                kind,
                directory / "selected_shape_sizzle",
                recipe=chosen,
                shots=shots,
                validation_shots=null_validation_shots,
                tolerance_phi_zz_rad=base.get("tolerance_phi_zz_rad"),
                bootstrap=bootstrap,
                recenter=recenter,
                budget=budget,
            )
            summary["sizzle_requalification"] = null_summary
        chosen["frozen_at"] = datetime.now().astimezone().isoformat()
        save_json(directory / "frozen_recipe.json", chosen)
        validation = short_gate_score(
            measurements,
            kind,
            chosen,
            directory / "independent_validation",
            validation=True,
            shots=validation_shots,
            budget=budget,
        )
        baseline_validation = short_gate_score(
            measurements,
            kind,
            base,
            directory / "baseline_comparison",
            validation=True,
            shots=validation_shots,
            budget=budget,
        )
        passed = (
            validation["ranking_score"] >= minimum_validation_score
            and validation["minimum_state_overlap"] >= minimum_validation_score
            and validation["population"]["minimum_population_agreement"]
            >= minimum_population_agreement
        )
        improvement = validation["score"] - baseline_validation["score"]
        error = np.hypot(
            validation["shot_standard_error"],
            baseline_validation["shot_standard_error"],
        )
        summary.update(
            qualified=bool(passed),
            status="qualified_shape" if passed else "independent_validation_failed",
            selected_trial=winner["index"],
            training_score=winner["score"]["score"],
            validation=validation,
            baseline_validation=baseline_validation,
            improvement=float(improvement),
            improvement_ci95_shots=[
                float(improvement - 1.96 * error),
                float(improvement + 1.96 * error),
            ],
            improvement_resolved_shot_only=bool(improvement - 1.96 * error > 0),
            optimizer_converged=bool(result.success),
            optimizer_message=str(result.message),
            evaluations=len(trials),
            budget=budget.report(),
            claim="fixed-family measured waveform optimization; independent state checks, not gate fidelity or measured gain",
        )
        save_json(directory / "optimization_summary.json", summary)
        if not passed:
            _reject("Frozen optimized SQUAD failed independent state/population checks")
        chosen.update(
            shape_optimization_directory=str(directory), shape_validation_passed=True
        )
        save_json(directory / "qualified_recipe.json", chosen)
    except Exception as error:
        summary.update(
            qualified=False, status="failed", error=str(error), budget=budget.report()
        )
        save_json(directory / "optimization_summary.json", summary)
        raise
    return chosen, summary
