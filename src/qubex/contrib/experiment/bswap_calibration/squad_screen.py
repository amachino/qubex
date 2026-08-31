"""
Bounded coupled SQUAD-design screening with a competing, frozen baseline.

This opt-in controller acquires through a caller-owned count port and writes
run-local evidence. It never connects, configures hardware, replaces the port's
recipes, or reloads an imported module. Coupled design scale/CD candidates test
a hypothetical gain design, not a measurement of delivered strong-drive gain.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .measurements import save_json
from .optimization import (
    QualificationError,
    ShotBudget,
    _hash_recipe,
    _null_pass,
    _refresh_phases,
    calibrate_sizzle,
    phase_cycle_zz,
    recenter_amplitude_frequency,
    short_gate_score,
)
from .pulses import make_squad_pulse, measured_phase_vectors


class ScreenIdentityError(RuntimeError):
    """The frozen calibration/measurement identity changed during screening."""


class ScreenTrainingBudgetExceeded(RuntimeError):
    """The preallocated search partition is exhausted; final reserve is untouched."""


class ScreenGlobalBudgetExceeded(RuntimeError):
    """An unexpected attempt would exceed the complete predeclared shot budget."""


def _baseline_failed() -> None:
    raise QualificationError(
        "Frozen baseline failed fresh current null/state validation"
    )


class _Partition(ShotBudget):
    def __init__(self, maximum: int, master: ShotBudget, *, training: bool) -> None:
        super().__init__(maximum)
        self.master, self.training = master, training

    def reserve(self, shots: int) -> None:
        if isinstance(shots, bool) or int(shots) != shots or shots <= 0:
            raise ValueError("Requested shots must be a positive integer")
        if self.requested + shots > self.maximum:
            error = (
                ScreenTrainingBudgetExceeded
                if self.training
                else ScreenGlobalBudgetExceeded
            )
            raise error(
                "Preallocated training partition exhausted"
                if self.training
                else "Final validation partition exceeded"
            )
        if self.master.requested + shots > self.master.maximum:
            raise ScreenGlobalBudgetExceeded("Global screen shot budget exceeded")
        self.master.reserve(shots)
        super().reserve(shots)

    def report(self) -> dict[str, int]:
        return {
            **super().report(),
            "global_requested_shots": self.master.requested,
            "global_maximum_shots": self.master.maximum,
        }


def _family(recipe: Any) -> dict[str, Any]:
    return {
        "duration_ns": recipe["duration_ns"],
        "ramp_ns": recipe.get("ramp_ns", 16.0),
        "window": deepcopy(recipe.get("window", {"type": "hann"})),
        "gate_start_ns": recipe["gate_start_ns"],
    }


def _port_identity(port: Any) -> str:
    pulses = {}
    for name in ("x90", "xpi"):
        collection = getattr(port, name, {})
        pulses[name] = {
            q: dict(
                sampling_period=float(p.sampling_period),
                duration=float(p.duration),
                samples_sha256=hashlib.sha256(
                    np.asarray(p.values, dtype=np.complex128).tobytes()
                ).hexdigest(),
            )
            for q, p in collection.items()
        }
    classifiers = {}
    for q, clf in getattr(port, "classifiers", {}).items():
        record: dict[str, Any] = {
            "type": type(clf).__module__ + "." + type(clf).__name__
        }
        for field in ("phase", "scale", "created_at", "means", "covariances"):
            if hasattr(clf, field):
                record[field] = getattr(clf, field)
        if hasattr(clf, "label_map"):
            record["label_map"] = {str(k): int(v) for k, v in clf.label_map.items()}
        model = getattr(clf, "model", None)
        record["predictor"] = {
            field: getattr(model, field)
            for field in (
                "weights_",
                "means_",
                "precisions_cholesky_",
                "precisions_",
                "covariances_",
                "cluster_centers_",
                "covariance_type",
                "n_features_in_",
            )
            if hasattr(model, field)
        }
        classifiers[q] = record
    return _hash_recipe(
        dict(
            session=port.session_id,
            rabi_scale=port.rabi_scale,
            references=port.references,
            targets=getattr(port, "targets", {}),
            qubits=port.qubits,
            pulses=pulses,
            classifiers=classifiers,
            recipes=port.recipes,
        )
    )


class _GuardedPort:
    def __init__(self, port: Any, kind: str, baseline: Any) -> None:
        self.port, self.kind, self.family = port, kind, _hash_recipe(_family(baseline))
        self.identity = _port_identity(port)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.port, name)

    def check(self, recipe: Any | None = None) -> None:
        if _port_identity(self.port) != self.identity:
            raise ScreenIdentityError(
                "Frozen K, pulse, classifier, frequency, session or port recipe changed"
            )
        if recipe is not None and _hash_recipe(_family(recipe)) != self.family:
            raise ScreenIdentityError(
                "Total duration, ramp, window or calibrated start changed"
            )

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        selected = kwargs.get("recipes", self.port.recipes)
        self.check(selected[self.kind])
        row = self.port.acquire(*args, **kwargs)
        self.check(selected[self.kind])
        counts = np.asarray(row["counts"])
        if (
            counts.shape != (4,)
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(counts != np.rint(counts))
            or counts.sum() != kwargs["shots"]
        ):
            raise ValueError("Screen acquisition returned malformed all-shot counts")
        return row


def _waveform(port: Any, recipe: Any, directory: Path) -> dict[str, Any]:
    pulse = make_squad_pulse(
        recipe,
        rabi_ghz_per_amplitude=port.rabi_scale,
        transition_frequency_ghz=port.references[port.qubits[0]],
    )
    values = np.asarray(pulse.values, dtype=np.complex128)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "envelope.npz", times_ns=pulse.times, iq_command=values
    )
    physical = float(
        2 * np.pi * (port.references[port.qubits[0]] - recipe["frequency_ghz"])
    )
    scale, cd = (
        float(recipe.get("design_delta_scale", 1)),
        float(recipe.get("cd_strength", 1)),
    )
    record = dict(
        fixed_K_rad_per_ns_per_command=float(2 * np.pi * port.rabi_scale),
        reference_minus_carrier_rad_per_ns=physical,
        design_delta_rad_per_ns=scale * physical,
        design_delta_scale=scale,
        cd_strength=cd,
        c_over_s=cd / scale,
        amplitude_command=float(recipe["amplitude"]),
        carrier_ghz=float(recipe["frequency_ghz"]),
        total_duration_ns=float(pulse.duration),
        ramp_ns=float(recipe.get("ramp_ns", 16)),
        sampling_period_ns=float(pulse.sampling_period),
        peak_abs_Q=float(np.max(abs(values.imag))),
        peak_complex_command=float(np.max(abs(values))),
        envelope_npz=str(directory / "envelope.npz"),
        samples_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        scope="unshifted base envelope before compiler carrier/frame phase; gain-design hypothesis, not measured gain",
        historical_squad_delta_control=recipe.get("squad_delta_control"),
        historical_delta_used_for_materialization=False,
    )
    save_json(directory / "waveform_manifest.json", record)
    return record


def _score_passes(score: Any, minimum_score: float, minimum_population: float) -> bool:
    numbers = [
        score[k]
        for k in (
            "score",
            "shot_standard_error",
            "ranking_score",
            "minimum_state_overlap",
        )
    ]
    numbers.append(score["population"]["minimum_population_agreement"])
    if not np.isfinite(numbers).all() or score["shot_standard_error"] < 0:
        raise ValueError("Screen score or uncertainty is nonfinite/invalid")
    return bool(
        score["ranking_score"] >= minimum_score
        and score["minimum_state_overlap"] >= minimum_score
        and score["population"]["minimum_population_agreement"] >= minimum_population
    )


def _comparison(candidate: Any, baseline: Any) -> dict[str, Any]:
    difference = float(candidate["score"] - baseline["score"])
    se = float(
        np.hypot(candidate["shot_standard_error"], baseline["shot_standard_error"])
    )
    interval = [difference - 1.96 * se, difference + 1.96 * se]
    return dict(
        improvement=difference,
        improvement_ci95_shots=interval,
        resolved_improvement=bool(interval[0] > 0),
        uncertainty="shot-only; no SPAM, seed-ensemble or drift systematics",
    )


def screen_squad_gain_hypotheses(
    measurements: Any,
    kind: str,
    directory: str | Path,
    *,
    candidate_pairs: Sequence[tuple[float, float]] | None = None,
    allow_endpoint_extension: bool = True,
    shots: int = 512,
    selection_shots: int = 2048,
    validation_shots: int = 2048,
    null_training_shots: int = 2048,
    null_validation_shots: int = 8192,
    max_total_shots: int = 4_000_000,
    minimum_validation_score: float = 0.65,
    minimum_population_agreement: float = 0.65,
    bootstrap: int = 400,
    null_ratio_grid: Sequence[float] = (0.0, 0.015, 0.03, 0.045, 0.06),
    max_null_refinements: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Screen bounded coupled SQUAD designs while retaining a verified baseline option.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Caller-owned current count port with a same-session qualified siZZle baseline.
    kind : str
        Independently calibrated bswap or sqrt_bswap family.
    directory : str or Path
        Fresh run-local evidence directory. Existing protocol files are refused.
    candidate_pairs : Sequence[tuple[float, float]] or None, optional
        (design_delta_scale, cd_strength). Default (1.3,1.3),(1.6,1.6).
        Explicit empty input validates only the baseline; one pair supports an
        independently calibrated full-gate transfer test. Baseline must be (1,1).
    allow_endpoint_extension : bool, optional
        With the two default candidates, allow (2,2) only when 1.6's ranking
        exceeds the baseline and 1.3, and sampled headroom passes.
    shots, selection_shots, validation_shots : int, optional
        Initial recenter/phase/score, fresh selection-score and final held-out
        score shots, default 512/2048/2048. Selection remeasurements are training.
    null_training_shots, null_validation_shots : int, optional
        Fixed-A/f siZZle training and independent/final ZZ shots, default 2048/8192.
    max_total_shots : int, optional
        Global cap including a protected final baseline/candidate reserve, default 4M.
    minimum_validation_score, minimum_population_agreement : float, optional
        Raw score/population criteria, both 0.65; neither is a gate fidelity.
    bootstrap : int, optional
        Existing phase-cycle bootstrap count, default 400.
    null_ratio_grid, max_null_refinements : Sequence[float], int, optional
        Predeclared amplitude-ratio grid and bounded refinements for new-shape
        fixed-control null searches. No old-shape ON map is reused.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Freshly verified recipe (possibly unchanged baseline) and full evidence.
        A training-budget fallback is labeled incomplete, never shape improvement.

    Raises
    ------
    QualificationError
        The frozen baseline no longer passes its fresh null or state checks.
    ScreenIdentityError, ScreenGlobalBudgetExceeded, TimeoutError, ValueError
        Identity, global-budget, deadline, waveform or malformed-data failure.
        These failures never become scientifically successful fallback.

    Notes
    -----
    Changed shapes are recentered and phase-calibrated before scoring. After
    positive fresh selection, run a precision fixed-A/f null, recenter once ON
    that new null, and re-null once if A/f moved. Freeze before final ZZ/score
    validation. Only that changed candidate or the original baseline can return;
    final validation cannot select an alternate candidate. K, duration, ramp,
    window, DRAG and classifiers stay fixed. Port recipes are never updated.
    Sampled manifests explicitly supersede inherited historical delta metadata.
    """
    if kind not in ("bswap", "sqrt_bswap"):
        raise ValueError("Unknown gate family")
    base: dict[str, Any] = deepcopy(measurements.recipes[kind])
    measured_phase_vectors(base)
    if (
        base.get("design_delta_scale", 1.0) != 1.0
        or base.get("cd_strength", 1.0) != 1.0
    ):
        raise ValueError(
            "This coupled screen requires an explicitly unit-design (1,1) baseline"
        )
    if (
        not base.get("null_shot_interval_passed", False)
        or base.get("phase_reference_session_id") != measurements.session_id
    ):
        raise ScreenIdentityError(
            "Baseline must be a qualified null in the current connection"
        )
    pairs = (
        [(1.3, 1.3), (1.6, 1.6)]
        if candidate_pairs is None
        else [tuple(map(float, pair)) for pair in candidate_pairs]
    )
    if len(set(pairs)) != len(pairs) or any(
        len(pair) != 2
        or not np.isfinite(pair).all()
        or pair[0] != pair[1]
        or not 1 < pair[0] <= 2
        for pair in pairs
    ):
        raise ValueError("Candidate pairs must be unique coupled s=c values in (1,2]")
    (
        shots,
        selection_shots,
        validation_shots,
        null_training_shots,
        null_validation_shots,
    ) = (
        ShotBudget(value).maximum
        for value in (
            shots,
            selection_shots,
            validation_shots,
            null_training_shots,
            null_validation_shots,
        )
    )
    ratios = np.asarray(null_ratio_grid, dtype=float)
    if (
        ratios.ndim != 1
        or len(ratios) < 2
        or not np.isfinite(ratios).all()
        or ratios[0] != 0
        or np.any(np.diff(ratios) <= 0)
        or not 0.03 <= ratios[-1] <= 0.1
    ):
        raise ValueError(
            "Null ratio grid must increase from zero through at least the .03 probe, within .1"
        )
    if (
        isinstance(max_null_refinements, bool)
        or int(max_null_refinements) != max_null_refinements
        or max_null_refinements < 1
    ):
        raise ValueError("max_null_refinements must be a positive integer")
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 2:
        raise ValueError("bootstrap must be an integer >=2")
    if (
        not 0 < minimum_validation_score <= 1
        or not 0 < minimum_population_agreement <= 1
    ):
        raise ValueError("Invalid predeclared score/population thresholds")
    master = ShotBudget(max_total_shots)
    reserve_each = 32 * null_validation_shots + 40 * validation_shots
    final_reserve = 2 * reserve_each
    if master.maximum <= final_reserve:
        raise ValueError(
            "Global budget must exceed the explicit two-recipe final validation reserve"
        )
    training = _Partition(master.maximum - final_reserve, master, training=True)
    validation = _Partition(final_reserve, master, training=False)
    port = _GuardedPort(measurements, kind, base)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "protocol.json").exists():
        raise FileExistsError("Use a fresh SQUAD screen directory")
    bases = 5 if kind == "sqrt_bswap" else 1
    recenter_typical = (42 * bases + 4 * (4 * bases + 4)) * shots
    recenter_maximum = (168 * bases + 4 * (4 * bases + 4)) * shots
    tolerance = float(
        base.get("tolerance_phi_zz_rad", 0.02 if kind == "sqrt_bswap" else 0.03)
    )
    if not 0 < tolerance < np.pi / 8:
        raise ValueError("Invalid baseline integrated-ZZ tolerance")
    if not _null_pass(base["null_validation"], tolerance, 0.02, 0.6):
        raise ScreenIdentityError(
            "Baseline lacks retained qualified ZZ confidence intervals"
        )
    summary: dict[str, Any] = dict(
        qualified=False,
        status="screening",
        kind=kind,
        retained_baseline=False,
        selected_shape_changed=False,
        selected_pair=None,
        resolved_improvement=False,
        screen_complete=bool(pairs),
        selection_reason=None,
        candidates=[],
        selection_training=None,
        final_validation=None,
        baseline_recipe=base,
        baseline_recipe_hash=_hash_recipe(base),
        fixed_K_rad_per_ns_per_command=2 * np.pi * port.rabi_scale,
        fixed_family=_family(base),
        candidate_pairs=pairs,
        allow_endpoint_extension=bool(allow_endpoint_extension),
        training_partition_limit=training.maximum,
        final_validation_reserve=final_reserve,
        global_shot_limit=master.maximum,
        projected_cost=dict(
            baseline_training=40 * shots if pairs else 0,
            per_changed_candidate_typical=recenter_typical + 76 * shots,
            per_changed_candidate_maximum=recenter_maximum + 76 * shots,
            selection_pair=80 * selection_shots if pairs else 0,
            fixed_control_null_minimum=300 * null_training_shots
            + 36 * null_validation_shots,
            fixed_control_null_maximum=(
                160 + 36 * len(null_ratio_grid) + 68 * max_null_refinements
            )
            * null_training_shots
            + 36 * null_validation_shots,
            post_null_recenter_typical=recenter_typical,
            post_null_recenter_maximum=recenter_maximum,
            final_per_recipe=reserve_each,
            baseline_only=reserve_each,
        ),
        claim="bounded gain-design hypothesis screen, not measured gain/global optimum/gate fidelity",
        fallback_policy="only frozen baseline after fresh validation; no alternate changed candidate",
        classifier_identity_scope="Qubex phase/scale/label map and fitted GMM/KMeans predictor arrays",
    )
    save_json(directory / "protocol.json", summary)
    chosen = None
    fallback = (
        "baseline_validated_screen_skipped"
        if not pairs
        else "baseline_retained_no_resolved_gain"
    )

    def checkpoint(stage: str) -> None:
        summary.update(
            current_stage=stage,
            budget=master.report(),
            training_budget=training.report(),
            validation_budget=validation.report(),
        )
        save_json(directory / "screen_summary.json", summary)
        save_json(
            directory / "candidates.json",
            dict(candidates=summary["candidates"], budget=master.report()),
        )

    def score(record: Any, where: Path, count: int, *, final: bool = False) -> Any:
        port.check(record)
        result = short_gate_score(
            port,
            kind,
            record,
            where,
            shots=count,
            validation=final,
            budget=validation if final else training,
        )
        _score_passes(result, minimum_validation_score, minimum_population_agreement)
        port.check(record)
        return result

    def null_search(record: Any, where: Path) -> dict[str, Any]:
        port.check(record)
        candidate, report = calibrate_sizzle(
            port,
            kind,
            where,
            recipe=record,
            shots=null_training_shots,
            validation_shots=null_validation_shots,
            tolerance_phi_zz_rad=tolerance,
            ratio_grid=null_ratio_grid,
            bootstrap=bootstrap,
            max_refinements=max_null_refinements,
            recenter=False,
            budget=training,
        )
        if not report["qualified"]:
            raise QualificationError("New-shape fixed-control null is unqualified")
        if not candidate.get("null_shot_interval_passed", False):
            raise ValueError(
                "Null helper reported success without qualified recipe evidence"
            )
        measured_phase_vectors(candidate)
        port.check(candidate)
        summary.setdefault("null_searches", []).append(
            dict(directory=str(where), report=report)
        )
        return candidate

    def final_check(record: Any, name: str) -> dict[str, Any]:
        port.check(record)
        result = phase_cycle_zz(
            port,
            kind,
            record,
            directory / name / "zz",
            shots=null_validation_shots,
            bootstrap=bootstrap,
            seed=7781 if name == "final_baseline" else 7782,
            budget=validation,
        )
        state_score = score(
            record, directory / name / "score", validation_shots, final=True
        )
        passed = _null_pass(result["estimate"], tolerance, 0.02, 0.6) and _score_passes(
            state_score, minimum_validation_score, minimum_population_agreement
        )
        return dict(null=result["estimate"], score=state_score, passed=bool(passed))

    try:
        summary["baseline_waveform"] = _waveform(
            port, base, directory / "baseline_waveform"
        )
        try:
            if pairs:
                checkpoint("baseline_training")
                baseline_score = score(base, directory / "baseline_training", shots)
                summary["baseline_training"] = baseline_score
                index = 0
                while index < len(pairs):
                    pair = pairs[index]
                    row: dict[str, Any] = dict(
                        index=index,
                        pair=list(pair),
                        design_delta_scale=pair[0],
                        cd_strength=pair[1],
                        status="calibrating",
                    )
                    summary["candidates"].append(row)
                    where = directory / "candidates" / f"{index:02d}"
                    candidate = {
                        **deepcopy(base),
                        "design_delta_scale": pair[0],
                        "cd_strength": pair[1],
                        "null_shot_interval_passed": False,
                        "shape_validation_passed": False,
                        "phase_status": "changed shape; local phases and null require fresh qualification",
                    }
                    checkpoint(f"candidate_{index}")
                    row["initial_waveform"] = _waveform(
                        port, candidate, where / "initial_waveform"
                    )
                    try:
                        candidate, _ = recenter_amplitude_frequency(
                            port,
                            kind,
                            candidate,
                            where / "recenter",
                            shots=shots,
                            budget=training,
                        )
                        port.check(candidate)
                        candidate = _refresh_phases(
                            port,
                            kind,
                            candidate,
                            where / "vz",
                            shots=shots,
                            budget=training,
                        )
                        port.check(candidate)
                        row.update(
                            recipe=candidate,
                            waveform=_waveform(
                                port, candidate, where / "scored_waveform"
                            ),
                            score=score(candidate, where / "score", shots),
                            status="scored",
                        )
                    except QualificationError as error:
                        row.update(
                            status="candidate_qualification_failed", error=str(error)
                        )
                    checkpoint(f"candidate_{index}_finished")
                    if (
                        index == 1
                        and allow_endpoint_extension
                        and pairs == [(1.3, 1.3), (1.6, 1.6)]
                    ):
                        earlier = summary["candidates"][0]
                        if row["status"] == earlier["status"] == "scored" and row[
                            "score"
                        ]["ranking_score"] > max(
                            baseline_score["ranking_score"],
                            earlier["score"]["ranking_score"],
                        ):
                            extension = {
                                **deepcopy(base),
                                "design_delta_scale": 2.0,
                                "cd_strength": 2.0,
                            }
                            try:
                                summary["endpoint_waveform"] = _waveform(
                                    port, extension, directory / "endpoint_headroom"
                                )
                            except ValueError as error:
                                if (
                                    str(error)
                                    != "Sampled SQUAD I+iQ exceeds the command headroom"
                                ):
                                    raise
                                summary["endpoint_extension_skipped"] = (
                                    "sampled complex headroom"
                                )
                            else:
                                pairs.append((2.0, 2.0))
                    index += 1
                pool = [
                    row
                    for row in summary["candidates"]
                    if row["status"] == "scored"
                    and _score_passes(
                        row["score"],
                        minimum_validation_score,
                        minimum_population_agreement,
                    )
                ]
                if pool:
                    best = max(pool, key=lambda row: row["score"]["ranking_score"])
                    candidate = deepcopy(best["recipe"])
                    checkpoint("second_stage_selection_training")
                    candidate_score = score(
                        candidate,
                        directory / "selection_training" / "candidate",
                        selection_shots,
                    )
                    base_score = score(
                        base,
                        directory / "selection_training" / "baseline",
                        selection_shots,
                    )
                    selection = dict(
                        candidate=candidate_score,
                        baseline=base_score,
                        role="second_stage_training",
                        selected_pair=best["pair"],
                        **_comparison(candidate_score, base_score),
                    )
                    summary["selection_training"] = selection
                    save_json(directory / "selection_training.json", selection)
                    if selection["resolved_improvement"] and _score_passes(
                        candidate_score,
                        minimum_validation_score,
                        minimum_population_agreement,
                    ):
                        checkpoint("selected_shape_fixed_control_null")
                        candidate = null_search(
                            candidate, directory / "selected_shape_sizzle"
                        )
                        before = (candidate["amplitude"], candidate["frequency_ghz"])
                        checkpoint("selected_shape_on_null_recenter_once")
                        candidate, _ = recenter_amplitude_frequency(
                            port,
                            kind,
                            candidate,
                            directory / "post_null_recenter",
                            shots=shots,
                            budget=training,
                        )
                        port.check(candidate)
                        changed = (
                            candidate["amplitude"],
                            candidate["frequency_ghz"],
                        ) != before
                        summary["post_null_amplitude_frequency_changed"] = changed
                        if changed:
                            candidate["null_shot_interval_passed"] = False
                            candidate["shape_validation_passed"] = False
                            checkpoint("one_bounded_renull_after_recenter")
                            candidate = null_search(
                                candidate, directory / "after_recenter_sizzle"
                            )
                        chosen = candidate
                else:
                    fallback = "baseline_retained_candidate_qualification_failure"
        except ScreenTrainingBudgetExceeded as error:
            chosen = None
            fallback = "baseline_retained_budget_limit"
            summary.update(training_failure=str(error), screen_complete=False)
        except QualificationError as error:
            chosen = None
            fallback = "baseline_retained_candidate_qualification_failure"
            summary["training_failure"] = str(error)

        port.check(base)
        frozen = {
            "baseline": dict(
                recipe=base,
                waveform=_waveform(port, base, directory / "frozen_baseline_waveform"),
            ),
            "candidate": None,
        }
        if chosen is not None:
            frozen["candidate"] = dict(
                recipe=chosen,
                waveform=_waveform(
                    port, chosen, directory / "frozen_candidate_waveform"
                ),
            )
        save_json(directory / "frozen_comparison.json", frozen)
        checkpoint("final_baseline_validation")
        baseline_validation = final_check(base, "final_baseline")
        final: dict[str, Any] = dict(
            baseline=baseline_validation,
            candidate=None,
            improvement=None,
            improvement_ci95_shots=None,
            candidate_accepted=False,
        )
        summary["final_validation"] = final
        save_json(directory / "final_validation.json", final)
        if not baseline_validation["passed"]:
            _baseline_failed()
        if chosen is not None:
            checkpoint("final_selected_candidate_validation")
            candidate_validation = final_check(chosen, "final_candidate")
            comparison = _comparison(
                candidate_validation["score"], baseline_validation["score"]
            )
            accepted = (
                candidate_validation["passed"] and comparison["resolved_improvement"]
            )
            final.update(
                candidate=candidate_validation,
                candidate_accepted=bool(accepted),
                **comparison,
            )
            save_json(directory / "final_validation.json", final)
        accepted = bool(final["candidate_accepted"])
        selected: dict[str, Any] = deepcopy(base)
        if chosen is not None and accepted:
            selected = deepcopy(chosen)
        selected_case = "candidate" if accepted else "baseline"
        selected_null = deepcopy(final[selected_case]["null"])
        prior_zz_model = {
            "zz_phase_rad": selected.get("zz_phase_rad"),
            "null_validation": deepcopy(selected.get("null_validation")),
            "local_phase_fit_zz_phase_rad": selected["phase_calibration"].get(
                "zz_phase_rad"
            ),
        }
        selected.update(
            zz_phase_rad=float(selected_null["zz_phase_rad"]),
            null_validation=selected_null,
            shape_screen_prior_zz_model=prior_zz_model,
            zz_model_source={
                "file": str(directory / "final_validation.json"),
                "case": selected_case,
                "role": "pre-benchmark calibration/qualification; no XEB observations used",
            },
            local_phase_fit_zz_retained_as_historical=True,
        )
        selected_pair = [
            float(selected.get("design_delta_scale", 1)),
            float(selected.get("cd_strength", 1)),
        ]
        reason = (
            "changed shape passed fresh null/score and shot-only improvement"
            if accepted
            else fallback
        )
        summary.update(
            qualified=True,
            status="changed_shape_validated" if accepted else fallback,
            retained_baseline=not accepted,
            selected_shape_changed=accepted,
            selected_pair=selected_pair,
            resolved_improvement=accepted,
            selection_reason=reason,
        )
        selected.update(
            shape_screen_directory=str(directory),
            shape_validation_passed=True,
            shape_screen_selected_pair=selected_pair,
            shape_screen_improvement_resolved_shot_only=accepted,
            shape_screen_status=summary["status"],
            shape_search_performed=bool(pairs),
        )
        port.check(selected)
        save_json(directory / "frozen_recipe.json", selected)
        save_json(directory / "qualified_recipe.json", selected)
        checkpoint("finished")
    except Exception as error:
        summary.update(
            qualified=False,
            status="failed",
            error=str(error),
            error_type=type(error).__name__,
        )
        checkpoint("failed")
        raise
    return selected, summary
