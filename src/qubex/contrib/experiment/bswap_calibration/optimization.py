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

from .local_optimization import (
    estimate_response_ridge,
    plan_frequency_extensions,
    propose_gp_point,
    ridge_scout_points,
    select_conservative_plateau,
)
from .measurements import save_json, zero_phase_recipe
from .pulses import ideal_circuit_unitary, make_squad_pulse, measured_phase_vectors
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


class _SmokePhaseIdentificationError(QualificationError):
    """A smoke phase fit is unidentified despite complete well-formed counts."""


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


def _recenter_score(
    measurements: Any,
    kind: str,
    recipe: Mapping[str, Any],
    directory: Path,
    label: str,
    *,
    shots: int,
    budget: ShotBudget,
) -> dict[str, Any]:
    bases = ("ZZ",) if kind == "bswap" else SQRT_BASES
    selected = _selected(measurements, kind, recipe, uncorrected=True)
    rabi_scale = measurements.rabi_scale
    reference_frequency = float(measurements.references[measurements.qubits[0]])
    scores, variances, rows = [], [], []
    for direction, prepared in enumerate((("0", "0"), ("1", "1"))):
        counts = []
        for basis in bases:
            row = _acquire(
                measurements,
                [_gate(kind)],
                directory,
                f"{label}_{direction}_{basis}",
                prepared=prepared,
                basis=basis,
                recipes=selected,
                shots=shots,
                budget=budget,
            )
            if measurements.rabi_scale != rabi_scale:
                raise ValueError(
                    "Measured Rabi scale changed during fixed-family recenter"
                )
            if (
                float(measurements.references[measurements.qubits[0]])
                != reference_frequency
            ):
                raise ValueError(
                    "Active GE reference changed during fixed-family recenter"
                )
            counts.append(row["counts"])
            rows.append(row)
        if kind == "bswap":
            score = counts[0][3 if direction == 0 else 0] / shots
            variance = max(score * (1 - score) / shots, 1 / (4 * shots**2))
        else:
            score, variance = sqrt_score(counts)
        scores.append(float(score))
        variances.append(float(variance))
    family = recipe.get("pulse_family", "Squad")
    result = {
        "amplitude": float(recipe["amplitude"]),
        "frequency_ghz": float(recipe["frequency_ghz"]),
        "pulse_family": family,
        "score": float(np.mean(scores)),
        "variance": float(np.sum(variances) / 4),
        "direction_scores": scores,
        "direction_variances": variances,
        "measurements": rows,
    }
    if family == "Squad":
        transition = float(measurements.references[measurements.qubits[0]])
        scale = float(recipe.get("design_delta_scale", 1.0))
        result.update(
            design_delta_scale=scale,
            design_delta_ghz=scale * (transition - float(recipe["frequency_ghz"])),
        )
    return result


def _recenter_design_context(
    measurements: Any,
    recipe: Mapping[str, Any],
    frequency_bounds_ghz: tuple[float, float],
) -> dict[str, Any]:
    """Freeze a SQUAD design detuning across a physical-carrier search."""
    family = recipe.get("pulse_family", "Squad")
    if family == "RaisedCosine":
        if "design_delta_scale" in recipe:
            raise ValueError(
                "RaisedCosine recenter cannot carry a SQUAD design_delta_scale"
            )
        return {
            "pulse_family": family,
            "design_delta_applicable": False,
            "transition_frequency_ghz": None,
            "frozen_design_delta_ghz": None,
        }
    if family != "Squad":
        raise ValueError(f"Unknown bSWAP pulse family: {family!r}")
    transition = float(measurements.references[measurements.qubits[0]])
    seed_frequency = float(recipe["frequency_ghz"])
    seed_scale = float(recipe.get("design_delta_scale", 1.0))
    values = np.asarray(
        [transition, seed_frequency, seed_scale, *frequency_bounds_ghz], dtype=float
    )
    if not np.isfinite(values).all() or seed_scale <= 0:
        raise ValueError("SQUAD recenter frequencies and design scale must be finite")
    seed_detuning = transition - seed_frequency
    if seed_detuning == 0:
        raise ValueError("SQUAD recenter seed design detuning cannot be zero")
    low_detuning = transition - frequency_bounds_ghz[0]
    high_detuning = transition - frequency_bounds_ghz[1]
    if low_detuning * high_detuning <= 0:
        raise ValueError(
            "SQUAD recenter frequency bounds cannot cross zero design detuning"
        )
    return {
        "pulse_family": family,
        "design_delta_applicable": True,
        "transition_frequency_ghz": transition,
        "frozen_design_delta_ghz": seed_scale * seed_detuning,
    }


def _materialize_recenter_candidate(
    recipe: Mapping[str, Any],
    amplitude: float,
    frequency_ghz: float,
    design_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy one carrier trial while preserving its SQUAD design waveform."""
    candidate = deepcopy(dict(recipe))
    candidate.update(amplitude=float(amplitude), frequency_ghz=float(frequency_ghz))
    if not design_context["design_delta_applicable"]:
        candidate.pop("design_delta_scale", None)
        return candidate
    transition = float(design_context["transition_frequency_ghz"])
    frozen_delta = float(design_context["frozen_design_delta_ghz"])
    denominator = transition - float(frequency_ghz)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("SQUAD recenter carrier gives zero design detuning")
    scale = frozen_delta / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SQUAD recenter carrier crosses the signed design detuning")
    candidate["design_delta_scale"] = float(scale)
    return candidate


def recenter_amplitude_frequency(
    measurements: Any,
    kind: str,
    recipe: Any,
    directory: str | Path,
    *,
    shots: int = 256,
    amplitude_step: float = 0.0005,
    frequency_span_mhz: float = 0.6,
    max_amplitude: float = 0.99,
    frequency_points: int = 7,
    frequency_search_half_width_mhz: float = 2.0,
    max_extension_rounds: int = 2,
    max_scout_points: int = 84,
    confirmation_shots: int | None = None,
    minimum_score_lower_bound: float = 0.65,
    maximum_confirmed_degradation: float = 0.02,
    maximum_population_error: float = 0.30,
    budget: ShotBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Recenter a narrow response ridge at fixed complete duration and ramp.

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
    amplitude_step : float, optional
        Separation of three command-amplitude rows, default 0.0005.
    frequency_span_mhz : float, optional
        Symmetric carrier span in MHz, default 0.6.
    max_amplitude : float, optional
        Maximum main command amplitude, default 0.99.
    frequency_points : int, optional
        Odd number of frequencies per row, at least seven, default seven.
    frequency_search_half_width_mhz : float, optional
        Hard carrier bounds about the supplied seed, default 2 MHz.
    max_extension_rounds, max_scout_points : int, optional
        Hard extension-round and total scout-point caps, default two and 84.
    confirmation_shots : int or None, optional
        Fresh shots per setting for both candidate and seed, default 4*shots.
    minimum_score_lower_bound : float, optional
        Minimum raw directional confirmation lower bound and root-plateau
        simultaneous lower bound, default 0.65. Not a gate-fidelity threshold.
    maximum_confirmed_degradation : float, optional
        Reject when the candidate-minus-seed upper 95% bound is below minus
        this margin, default 0.02. This is not a proof of noninferiority.
    maximum_population_error : float, optional
        Maximum four-state distribution error on fresh shots, default 0.30.
    budget : ShotBudget or None, optional
        Shared requested-shot budget.

    Returns
    -------
    tuple[dict, dict]
        Recentered recipe and fit/population evidence.

    Raises
    ------
    QualificationError
        The bounded ridge, root plateau, confirmation, or shot budget fails.

    Notes
    -----
    Acquires a three-row scout, extends only missing frequency coverage, and
    ranks measured points using ridge coordinates after ridge qualification.
    If all three row peaks are interior but the linear ridge is rejected,
    an explicit physical-coordinate GP instead ranks within the observed
    control box, preserving the rejected ridge as diagnostic evidence. The
    ridge-mode and five-point local interpolation reduced-chi-square <=5
    gates are unchanged. A high unresolved root-coherence plateau retains the
    independent seed, without qualifying a ridge or allowing a GP. Fresh
    candidate/seed and population checks never refit the selection. Duration,
    ramp, signed SQUAD design detuning, CD, window, cancellation controls and
    measured K stay fixed; only command amplitude and physical carrier change.
    For SQUAD, design_delta_scale is therefore carrier-adaptive so that the
    materialized I+iQ waveform is unchanged within each amplitude row. An
    I-only RaisedCosine recipe has no design scale. Later phase, ZZ and held-out
    benchmark gates remain necessary. No score here is a gate-fidelity estimate.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _gate(kind)
    recipe = deepcopy(recipe)
    shots = ShotBudget(shots).maximum
    confirmation_shots = ShotBudget(
        4 * shots if confirmation_shots is None else confirmation_shots
    ).maximum
    for value, name, lower in (
        (frequency_points, "frequency_points", 7),
        (max_extension_rounds, "max_extension_rounds", 0),
        (max_scout_points, "max_scout_points", 3 * frequency_points),
    ):
        if isinstance(value, bool) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    controls = np.asarray(
        [
            frequency_search_half_width_mhz,
            minimum_score_lower_bound,
            maximum_confirmed_degradation,
            maximum_population_error,
        ],
        dtype=float,
    )
    if (
        not np.isfinite(controls).all()
        or frequency_points % 2 != 1
        or frequency_search_half_width_mhz < frequency_span_mhz
        or not 0 < minimum_score_lower_bound <= 1
        or not 0 <= maximum_confirmed_degradation <= 1
        or not 0 <= maximum_population_error <= 1
    ):
        raise ValueError("Invalid recenter bounds or confirmation thresholds")
    bases = ("ZZ",) if kind == "bswap" else SQRT_BASES
    maximum_shots = (
        2 * max_scout_points * len(bases) * shots
        + (4 * len(bases) + 4) * confirmation_shots
    )
    budget = budget or ShotBudget(maximum_shots)
    f0 = float(recipe["frequency_ghz"])
    frequency_bounds = (
        f0 - frequency_search_half_width_mhz / 1000,
        f0 + frequency_search_half_width_mhz / 1000,
    )
    design_context = _recenter_design_context(measurements, recipe, frequency_bounds)
    scout = ridge_scout_points(
        recipe["amplitude"],
        f0,
        amplitude_step=amplitude_step,
        frequency_half_width_mhz=frequency_span_mhz,
        frequency_points=frequency_points,
        amplitude_bounds=(0.01, max_amplitude),
        frequency_bounds_ghz=frequency_bounds,
    )
    fixed_family = {
        key: deepcopy(value)
        for key, value in recipe.items()
        if key
        not in (
            "amplitude",
            "frequency_ghz",
            "design_delta_scale",
            "phase_calibration",
        )
    }
    plan = {
        "scout": scout,
        "initial_points": len(scout["points"]),
        "amplitude_step": amplitude_step,
        "max_scout_points": max_scout_points,
        "max_extension_rounds": max_extension_rounds,
        "frequency_bounds_ghz": frequency_bounds,
        "shots": shots,
        "confirmation_shots": confirmation_shots,
        "maximum_shots": maximum_shots,
        "fixed_family": fixed_family,
        **design_context,
        "fixed_rabi_scale": measurements.rabi_scale,
        "fixed_family_fingerprint": _hash_recipe(
            {"fixed_family": fixed_family, "design_context": design_context}
        ),
        "minimum_score_lower_bound": minimum_score_lower_bound,
        "maximum_confirmed_degradation": maximum_confirmed_degradation,
        "maximum_population_error": maximum_population_error,
        "gp_acquisition_points": 0,
    }
    save_json(directory / "recenter_plan.json", plan)

    def require_fixed_scale() -> None:
        if measurements.rabi_scale != plan["fixed_rabi_scale"]:
            raise ValueError("Measured Rabi scale changed during fixed-family recenter")
        if design_context["design_delta_applicable"] and (
            float(measurements.references[measurements.qubits[0]])
            != design_context["transition_frequency_ghz"]
        ):
            raise ValueError(
                "Active GE reference changed during fixed-design-delta recenter"
            )

    observations: list[dict[str, Any]] = []
    extensions = []
    points = scout["points"]
    fit: dict[str, Any] = {}
    for round_index in range(max_extension_rounds + 1):
        for index in np.random.default_rng(891 + round_index).permutation(len(points)):
            require_fixed_scale()
            amplitude, frequency = points[index]
            candidate = _materialize_recenter_candidate(
                recipe, amplitude, frequency, design_context
            )
            observations.append(
                _recenter_score(
                    measurements,
                    kind,
                    candidate,
                    directory,
                    f"map_{len(observations):03d}",
                    shots=shots,
                    budget=budget,
                )
            )
            save_json(directory / "local_observations.json", observations)
            design_arrays = {}
            if design_context["design_delta_applicable"]:
                design_arrays = {
                    "design_delta_scales": [
                        r["design_delta_scale"] for r in observations
                    ],
                    "design_delta_ghz": [r["design_delta_ghz"] for r in observations],
                    "frozen_design_delta_ghz": design_context[
                        "frozen_design_delta_ghz"
                    ],
                }
            np.savez_compressed(
                directory / "local_map_partial.npz",
                amplitudes=[r["amplitude"] for r in observations],
                frequencies_ghz=[r["frequency_ghz"] for r in observations],
                scores=[r["direction_scores"] for r in observations],
                variances=[r["direction_variances"] for r in observations],
                **design_arrays,
            )
        values = tuple(
            np.asarray([row[key] for row in observations])
            for key in (
                "amplitude",
                "frequency_ghz",
                "score",
                "variance",
            )
        )
        ridge = estimate_response_ridge(
            *values,
            reference_amplitude=recipe["amplitude"],
            frequency_bounds_ghz=frequency_bounds,
            maximum_reduced_chi2=5.0,
        )
        plateau = select_conservative_plateau(
            *values,
            gate_kind=kind,
            seed_amplitude=recipe["amplitude"],
            seed_frequency_ghz=f0,
            minimum_lower_confidence=minimum_score_lower_bound,
            frequency_neighborhood_mhz=frequency_span_mhz,
        )
        physical_gp_allowed = bool(
            not ridge["qualified"]
            and not plateau["accepted"]
            and len(ridge["rows"]) >= 3
            and all(row["interior_peak"] for row in ridge["rows"])
            and np.isfinite(ridge.get("reduced_chi2", np.nan))
            and ridge["reduced_chi2"] > 5.0
        )
        fit = {
            "ridge": ridge,
            "plateau": plateau,
            "qualified_ridge": ridge["qualified"],
            "gp_allowed": bool(ridge["qualified"] or physical_gp_allowed),
            "coordinate_mode": "ridge"
            if ridge["qualified"]
            else "physical"
            if physical_gp_allowed
            else None,
            "extensions": extensions,
            "observed_points": len(observations),
            "reduced_chi2": ridge.get("reduced_chi2"),
            "claim": "raw response selection, not Hamiltonian resonance or gate fidelity",
        }
        save_json(directory / "local_fit.json", fit)
        if fit["gp_allowed"] or plateau["accepted"]:
            break
        extension = plan_frequency_extensions(
            *values,
            ridge=ridge,
            frequency_bounds_ghz=frequency_bounds,
            round_index=round_index,
            max_rounds=max_extension_rounds,
            frequency_points=frequency_points,
            frequency_half_width_mhz=frequency_span_mhz,
        )
        remaining = max_scout_points - len(observations)
        points = extension["points"][:remaining]
        extension["acquired_point_plan"] = points
        extension["limited_by_point_cap"] = len(points) < len(extension["points"])
        extensions.append(extension)
        save_json(directory / "local_fit.json", fit)
        if not points:
            _reject(
                "No qualified response ridge or conservative root plateau within the declared budget"
            )
    record = _materialize_recenter_candidate(
        recipe, recipe["amplitude"], recipe["frequency_ghz"], design_context
    )
    if fit["gp_allowed"]:
        ranking = propose_gp_point(
            *values,
            ridge=fit["ridge"],
            coordinate_mode=fit["coordinate_mode"],
            amplitude_bounds=(float(values[0].min()), float(values[0].max())),
            frequency_bounds_ghz=frequency_bounds,
            frequency_half_width_mhz=frequency_span_mhz,
            amplitude_scale=amplitude_step,
            candidates=np.column_stack(values[:2]),
            allow_repeats=True,
        )
        selected = ranking["best_observed"]
        if not np.isfinite(selected["ranking_score"]):
            _reject("Nonfinite response-ridge candidate ranking")
        record = _materialize_recenter_candidate(
            recipe,
            selected["amplitude"],
            selected["frequency_ghz"],
            design_context,
        )
        fit.update(
            selection=f"measured {fit['coordinate_mode']}-coordinate GP lower-bound incumbent",
            ranking=ranking,
        )
    else:
        fit["selection"] = "retain independent seed: conservative root plateau, no GP"
    save_json(directory / "local_fit.json", fit)
    checks = {}
    seed_record = _materialize_recenter_candidate(
        recipe, recipe["amplitude"], recipe["frequency_ghz"], design_context
    )
    for name, candidate in (("seed", seed_record), ("candidate", record)):
        require_fixed_scale()
        checks[name] = _recenter_score(
            measurements,
            kind,
            candidate,
            directory / "confirmation",
            name,
            shots=confirmation_shots,
            budget=budget,
        )
    candidate_check, seed_check = checks["candidate"], checks["seed"]
    lower = np.asarray(candidate_check["direction_scores"]) - 1.96 * np.sqrt(
        candidate_check["direction_variances"]
    )
    difference_upper = float(
        candidate_check["score"]
        - seed_check["score"]
        + 1.96 * np.sqrt(candidate_check["variance"] + seed_check["variance"])
    )
    confirmation = {
        **checks,
        "minimum_direction_lower_bound": float(lower.min()),
        "candidate_minus_seed_upper95": difference_upper,
        "passed": bool(
            lower.min() >= minimum_score_lower_bound
            and difference_upper >= -maximum_confirmed_degradation
        ),
        "claim": "fresh raw-score checks; no noninferiority or gate-fidelity claim",
    }
    save_json(directory / "confirmation.json", confirmation)
    if not confirmation["passed"]:
        _reject("Independent recenter score confirmation failed")
    require_fixed_scale()
    population = _population_check(
        measurements,
        kind,
        record,
        directory / "confirmation",
        shots=confirmation_shots,
        budget=budget,
    )
    if population["minimum_population_agreement"] < 1 - maximum_population_error:
        _reject("Independent recenter population check failed")
    require_fixed_scale()
    record["local_refinement_directory"] = str(directory)
    save_json(directory / "recentered_recipe.json", record)
    return record, {
        "fit": fit,
        "plan": plan,
        "confirmation": confirmation,
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
    smoke_mode: bool = False,
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
        if smoke_mode:
            raise _SmokePhaseIdentificationError(
                f"Measured smoke phase calibration is unidentified: {error}"
            ) from error
        raise QualificationError(
            f"Measured phase calibration failed: {error}"
        ) from error
    if fit["coherence_residual_rms"] > 0.08 and not smoke_mode:
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
    if smoke_mode:
        record.update(
            smoke_only=True,
            qualified=False,
            scientific_qualified=False,
            null_shot_interval_passed=False,
            shape_validation_passed=False,
            phase_model_qualified=bool(fit["coherence_residual_rms"] <= 0.08),
            phase_status="finite measured smoke fit; scientific qualification deferred",
        )
    save_json(directory / "phase_calibration.json", record)
    return record


def _smoke_recipe(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    result.update(
        smoke_only=True,
        qualified=False,
        scientific_qualified=False,
        null_shot_interval_passed=False,
        shape_validation_passed=False,
    )
    return result


def _invalid_smoke_input(message: str) -> NoReturn:
    raise ValueError(message)


def _smoke_candidate_workflow(
    measurements: Any,
    kind: str,
    base: Mapping[str, Any],
    changed: Mapping[str, Any],
    directory: Path,
    *,
    purpose: str,
    modified_parameters: Mapping[str, float],
    shots: int,
    validation_shots: int,
    bootstrap: int,
    budget: ShotBudget,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise two exact waveforms without manufacturing a qualified calibration."""
    summary_name = (
        "sizzle_summary.json" if purpose == "sizzle" else "optimization_summary.json"
    )
    candidates: list[dict[str, Any]] = []
    coverage: dict[str, Any] = dict(
        planned_candidates=2,
        acquired_candidates=0,
        changed_waveform_acquired=False,
        changed_waveform_returned=False,
        skipped_full_optimization=True,
        independent_qualification_performed=False,
        phase_cycle_settings_per_candidate=32,
        phase_fit_settings_per_candidate=36,
        population_settings_per_candidate=4,
        unexercised_coverage=[],
    )
    summary: dict[str, Any] = dict(
        kind=kind,
        smoke_only=True,
        qualified=False,
        scientific_qualified=False,
        status="smoke_running",
        purpose=purpose,
        candidates=candidates,
        coverage=coverage,
        fixed_duration_ns=base["duration_ns"],
        fixed_ramp_ns=base.get("ramp_ns", 16.0),
        session_id=measurements.session_id,
        shots_per_setting=shots,
        population_shots_per_setting=validation_shots,
        claim="bounded workflow smoke with real counts; not a ZZ null, waveform optimum, or gate fidelity",
    )
    save_json(directory / "protocol.json", summary)
    try:
        # A retained seed must have real supplied phases; never synthesize them.
        measured_phase_vectors(base)
        waveform_hashes = []
        for record in (base, changed):
            pulse = make_squad_pulse(
                record,
                rabi_ghz_per_amplitude=measurements.rabi_scale,
                transition_frequency_ghz=measurements.references[
                    measurements.qubits[0]
                ],
            )
            ratio = float(record.get("cancel_amplitude_ratio", 0.0))
            phase = float(record.get("cancel_phase_rad", 0.0))
            if not np.isfinite([ratio, phase]).all() or not 0 <= ratio <= 1:
                _invalid_smoke_input("Invalid cancellation waveform in smoke candidate")
            samples = np.stack(
                [pulse.values, ratio * pulse.values * np.exp(1j * phase)]
            )
            waveform_hashes.append(hashlib.sha256(samples.tobytes()).hexdigest())
        if waveform_hashes[0] == waveform_hashes[1]:
            _invalid_smoke_input(
                "Smoke changed candidate must change actual sampled waveform"
            )

        for index, candidate in enumerate((base, changed)):
            trial_dir = directory / f"smoke_candidate_{index}"
            row: dict[str, Any] = dict(
                index=index,
                changed_waveform=index == 1,
                modified_parameters=dict(modified_parameters) if index else {},
                status="started",
                phase_identified=False,
                phase_model_qualified=False,
                phase_fit=None,
                waveform_sha256=waveform_hashes[index],
            )
            candidates.append(row)
            cycle = phase_cycle_zz(
                measurements,
                kind,
                candidate,
                trial_dir / "zz",
                shots=shots,
                bootstrap=bootstrap,
                seed=17100 + index,
                budget=budget,
            )
            row.update(
                zz_estimate=cycle["estimate"],
                phase_cycle_file=str(trial_dir / "zz/phase_cycle.json"),
            )
            coverage["acquired_candidates"] += 1
            if index:
                coverage["changed_waveform_acquired"] = True
            try:
                refreshed = _refresh_phases(
                    measurements,
                    kind,
                    candidate,
                    trial_dir / "phase",
                    shots=shots,
                    budget=budget,
                    smoke_mode=True,
                )
                measured_phase_vectors(refreshed)
                if not np.isfinite(float(refreshed["zz_phase_rad"])):
                    raise ValueError("Nonfinite measured smoke ZZ phase")
                row.update(
                    status="finite_phase_fit",
                    phase_identified=True,
                    phase_model_qualified=refreshed["phase_model_qualified"],
                    phase_fit=deepcopy(refreshed["phase_calibration"]),
                    recipe=_smoke_recipe(refreshed),
                )
                population_recipe = refreshed
            except _SmokePhaseIdentificationError as error:
                # This exception is specific to the final phase solve. Budget,
                # deadline, count-schema and waveform failures are never caught.
                incomplete = _smoke_recipe(candidate)
                for field in (
                    "phase_calibration",
                    "pre_vz_rad",
                    "post_vz_rad",
                    "zz_phase_rad",
                ):
                    incomplete.pop(field, None)
                incomplete["phase_status"] = "unidentified; diagnostic waveform only"
                row.update(
                    status="phase_unidentified",
                    phase_error=str(error),
                    recipe=incomplete,
                )
                population_recipe = zero_phase_recipe(candidate)
                print(
                    f"SMOKE_QUALITY_WARNING: candidate {index} has no identifiable new phase fit",
                    flush=True,
                )
            row["population"] = _population_check(
                measurements,
                kind,
                population_recipe,
                trial_dir / "population",
                shots=validation_shots,
                budget=budget,
            )
            save_json(
                directory / "smoke_candidates.json",
                {"candidates": candidates, "budget": budget.report()},
            )

        if candidates[1]["phase_identified"]:
            selected_index = 1
            chosen = _smoke_recipe(candidates[1]["recipe"])
            coverage["changed_waveform_returned"] = True
            selection = (
                "changed waveform with finite measured phases; smoke coverage only"
            )
        else:
            if "zz_phase_rad" not in base or not np.isfinite(
                float(base["zz_phase_rad"])
            ):
                _reject(
                    "No unchanged seed with supplied measured phases and a finite frozen ZZ phase is available"
                )
            selected_index = 0
            chosen = _smoke_recipe(base)
            coverage["unexercised_coverage"] = [
                "changed-waveform downstream compilation and benchmarking"
            ]
            if purpose == "sizzle":
                coverage["unexercised_coverage"].append("new siZZle-ON downstream path")
            selection = "unchanged seed with supplied measured phases retained; changed pulse phases unidentified"
            print(
                "SMOKE_UNCHANGED_SEED: changed-waveform downstream coverage remains untested",
                flush=True,
            )
        chosen.update(
            smoke_protocol=purpose,
            smoke_directory=str(directory),
            smoke_coverage=deepcopy(coverage),
            frozen_at=datetime.now().astimezone().isoformat(),
            status="smoke_only; not scientifically qualified",
        )
        summary.update(
            status="smoke_only",
            evaluations=len(candidates),
            selected_trial=selected_index,
            selection=selection,
            budget=budget.report(),
        )
        save_json(directory / "frozen_recipe.json", chosen)
        save_json(directory / "provisional_recipe.json", chosen)
        save_json(directory / summary_name, summary)
        print(
            "SMOKE_QUALITY_WARNING: only low-shot workflow coverage; no null or fidelity qualification",
            flush=True,
        )
    except Exception as error:
        summary.update(
            status="failed",
            error=str(error),
            error_type=type(error).__name__,
            budget=budget.report(),
        )
        save_json(
            directory / "smoke_candidates.json",
            {"candidates": candidates, "budget": budget.report()},
        )
        save_json(directory / summary_name, summary)
        raise
    return chosen, summary


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
    smoke_mode: bool = False,
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
    smoke_mode : bool, optional
        Exercise the seed and one bounded changed tone at low caller-selected
        shots. Full 32-setting cycles, measured phases and populations are retained,
        but optimization/null qualification is skipped and every output remains
        smoke-only/unqualified. No `qualified_recipe.json` is written.

    Returns
    -------
    tuple[dict, dict]
        Independently qualified recipe and complete acceptance summary.
        Smoke mode instead returns an unqualified provisional recipe and
        explicit workflow-coverage summary.

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
    if smoke_mode:
        ratio = (
            probe_ratio
            if not np.isclose(base.get("cancel_amplitude_ratio", 0.0), probe_ratio)
            else probe_ratio / 2
        )
        changed = {
            **deepcopy(base),
            "cancel_amplitude_ratio": float(ratio),
            "cancel_phase_rad": float(base.get("cancel_phase_rad", 0.0)),
        }
        return _smoke_candidate_workflow(
            measurements,
            kind,
            base,
            changed,
            directory,
            purpose="sizzle",
            modified_parameters={"cancel_amplitude_ratio": float(ratio)},
            shots=shots,
            validation_shots=validation_shots,
            bootstrap=bootstrap,
            budget=budget,
        )
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
    smoke_mode: bool = False,
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
    smoke_mode : bool, optional
        Exercise exactly two finite waveforms: the seed and a bounded design-delta
        perturbation. Acquire full ZZ cycles, actual phase fits and populations,
        skipping the optimizer/recenter and independent qualification. Return a
        smoke-only provisional recipe; never write `qualified_recipe.json`.

    Returns
    -------
    tuple[dict, dict]
        Frozen independently validated recipe and optimization/comparison record.
        Smoke mode instead returns an unqualified provisional recipe and
        explicit workflow-coverage summary.

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
    if smoke_mode:
        scale = float(x0[0] + (0.05 if x0[0] <= 1.55 else -0.05))
        changed = {**deepcopy(base), "design_delta_scale": scale}
        return _smoke_candidate_workflow(
            measurements,
            kind,
            base,
            changed,
            directory,
            purpose="squad",
            modified_parameters={"design_delta_scale": scale},
            shots=shots,
            validation_shots=validation_shots,
            bootstrap=bootstrap,
            budget=budget,
        )
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
