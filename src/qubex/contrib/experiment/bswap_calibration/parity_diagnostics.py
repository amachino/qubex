"""
Failure-only, fixed-waveform odd-sector diagnostics with emitted-pulse checks.

This module does not connect, configure, recalibrate, select a gate, or modify
the caller's recipes. A caller may explicitly acquire this diagnostic after
failed admission checks; the result never overrides that original failure.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .measurements import save_json
from .pulses import compile_campaign, ideal_circuit_unitary
from .tomography import BASES, PAULI, density_from_counts, state_vector

VARIANTS = ("standard", "parity", "idle")
PREPARED = ("0", "1")


def _hash(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=encode).encode()
    ).hexdigest()


def _samples(pulse: Any) -> NDArray[np.complex128]:
    return np.asarray(pulse, dtype=np.complex128)


def _identity(port: Any) -> dict[str, Any]:
    pulses = {
        name: {
            q: dict(
                sampling_period_ns=float(p.sampling_period),
                duration_ns=float(p.duration),
                samples_sha256=hashlib.sha256(_samples(p.values).tobytes()).hexdigest(),
            )
            for q, p in getattr(port, name).items()
        }
        for name in ("x90", "xpi")
    }
    classifiers = {}
    for q, classifier in port.classifiers.items():
        model = getattr(classifier, "model", None)
        classifiers[q] = dict(
            type=type(classifier).__module__ + "." + type(classifier).__name__,
            fields={
                key: deepcopy(getattr(classifier, key))
                for key in ("phase", "scale", "means", "covariances")
                if hasattr(classifier, key)
            },
            label_map={
                str(k): int(v) for k, v in getattr(classifier, "label_map", {}).items()
            },
            predictor={
                key: deepcopy(getattr(model, key))
                for key in (
                    "weights_",
                    "means_",
                    "covariances_",
                    "precisions_",
                    "precisions_cholesky_",
                    "cluster_centers_",
                    "covariance_type",
                )
                if hasattr(model, key)
            },
        )
    return dict(
        session_id=port.session_id,
        recipes=deepcopy(port.recipes),
        qubits=tuple(port.qubits),
        drive_label=port.drive_label,
        cancel_label=port.cancel_label,
        rabi_scale_ghz_per_command=port.rabi_scale,
        references_ghz=deepcopy(port.references),
        targets_ghz=deepcopy(port.targets),
        backend_preamble_ns=port.backend_preamble_ns,
        shot_interval_ns=port.shot_interval_ns,
        production_drag=pulses,
        classifiers=classifiers,
        assignment_source=port.assignment_source,
    )


def _compile(
    port: Any, gates: Any, recipes: Any, basis: str
) -> tuple[Any, dict[str, Any]]:
    return compile_campaign(
        gates,
        recipes=recipes,
        qubits=port.qubits,
        drive_label=port.drive_label,
        cancel_label=port.cancel_label,
        target_frequencies_ghz=port.targets,
        reference_frequencies_ghz=port.references,
        rabi_ghz_per_amplitude=port.rabi_scale,
        x90=port.x90,
        xpi=port.xpi,
        backend_preamble_ns=port.backend_preamble_ns,
        prepared=PREPARED,
        basis=basis,
        initial_frame=(0.0, 0.0),
    )


def _same(a: Any, b: Any, context: str) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Diagnostic preflight: invalid {context}")
    difference = float(np.max(np.abs(a - b), initial=0.0))
    if difference > 2e-11:
        raise ValueError(f"Diagnostic preflight: {context} differs ({difference:g})")
    return difference


def _preflight(
    port: Any, recipes: Any, directory: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = recipes["bswap"]
    if float(recipe.get("cancel_amplitude_ratio", 0.0)) <= 0:
        raise ValueError(
            "Diagnostic requires a nonzero cancellation tone to verify its sign"
        )
    duration = 4 * float(recipe["duration_ns"])
    standard = ["BSWAP"] * 4
    parity = ["BSWAP", ("VZ", np.pi, np.pi), "BSWAP", ("VZ", -np.pi, -np.pi)] * 2
    _, standard_zz = _compile(port, standard, recipes, "ZZ")
    frame = standard_zz["final_frame_rad"]
    variants = dict(
        standard=standard,
        parity=parity,
        idle=[("IDLE", duration), ("VZ", *frame)],
    )
    zz = float(recipe["zz_phase_rad"])
    if not np.isfinite(zz):
        raise ValueError("Frozen ZZ target must be finite")
    unitary = ideal_circuit_unitary(standard, zz_phases={"bswap": zz})
    comparison = ideal_circuit_unitary(parity, zz_phases={"bswap": zz})
    phase = np.angle(np.trace(unitary.conj().T @ comparison))
    ideal_difference = _same(
        unitary, comparison * np.exp(-1j * phase), "ideal parity circuit"
    )
    state = state_vector(PREPARED)
    idle_state = ideal_circuit_unitary(variants["idle"]) @ state
    driven_state = unitary @ state
    state_phase = np.angle(np.vdot(driven_state, idle_state))
    _same(driven_state, idle_state * np.exp(-1j * state_phase), "ideal input-01 output")
    labels = (*port.qubits, port.drive_label, port.cancel_label)
    reports, arrays = {}, {}
    max_ge = max_tone = 0.0
    for basis in BASES:
        emitted, basis_reports = {}, {}
        for name, gates in variants.items():
            schedule, report = _compile(port, gates, recipes, basis)
            basis_reports[name] = report
            emitted[name] = {
                label: _samples(
                    schedule.get_sequence(label).get_values(apply_frame_shifts=True)
                )
                for label in labels
            }
            for index, label in enumerate(labels):
                arrays[f"{basis}_{name}_channel_{index}"] = emitted[name][label]
        reference = basis_reports["standard"]
        events = [e for e in reference["events"] if e["kind"] == "bswap"]
        other_events = [
            e for e in basis_reports["parity"]["events"] if e["kind"] == "bswap"
        ]
        if len(events) != 4 or len(other_events) != 4:
            raise ValueError("Diagnostic requires exactly four full bSWAP pulses")
        start = reference["preparation_window_ns"]
        for name in VARIANTS:
            _same(
                reference["duration_ns"],
                basis_reports[name]["duration_ns"],
                "complete duration",
            )
            _same(
                start,
                basis_reports[name]["preparation_window_ns"],
                "preparation origin",
            )
            for label in port.qubits:
                max_ge = max(
                    max_ge,
                    _same(
                        emitted["standard"][label],
                        emitted[name][label],
                        "GE preparation/analysis",
                    ),
                )
        for index, (event, other) in enumerate(zip(events, other_events, strict=True)):
            _same(
                event["start_ns"],
                start + index * float(recipe["duration_ns"]),
                "post-preparation gate placement",
            )
            _same(event["start_ns"], other["start_ns"], "parity pulse placement")
            _same(event["duration_ns"], other["duration_ns"], "parity pulse duration")
            first = round(event["start_ns"] / 2)
            last = first + round(event["duration_ns"] / 2)
            sign = 1 if index % 2 == 0 else -1
            for label in (port.drive_label, port.cancel_label):
                values = emitted["standard"][label][first:last]
                if np.max(np.abs(values), initial=0.0) <= 1e-12:
                    raise ValueError(
                        "Expected a nonzero tone at every driven occurrence"
                    )
                max_tone = max(
                    max_tone,
                    _same(
                        sign * values,
                        emitted["parity"][label][first:last],
                        f"{label} occurrence {index + 1} sign",
                    ),
                )
                _same(
                    emitted["idle"][label],
                    np.zeros_like(emitted["idle"][label]),
                    "idle drive blank",
                )
        reports[basis] = basis_reports
    np.savez_compressed(directory / "preflight_waveforms.npz", **arrays)
    result = dict(
        passed=True,
        initial_logical_frame_rad=[0.0, 0.0],
        supplied_measured_phase_calibration=deepcopy(recipe["phase_calibration"]),
        gate_body_duration_ns=duration,
        expected_tone_signs=[1, -1, 1, -1],
        maximum_ge_difference=max_ge,
        maximum_tone_difference=max_tone,
        ideal_unitary_difference=ideal_difference,
        fixed_zz_phase_rad=zz,
        idle_terminal_frame_rad=frame,
        idle_scope="Post-preparation wait with frame-matched analysis; ideal input 01 only, not general unitary equivalence",
        channel_order=list(labels),
        sampling_period_ns=2.0,
        compiled_reports=reports,
        waveform_file=str(directory / "preflight_waveforms.npz"),
    )
    save_json(directory / "preflight.json", result)
    return variants, result


def _coherence_weights(row: int, column: int) -> NDArray[np.complex128]:
    weights = np.empty((9, 4), dtype=complex)
    for index, basis in enumerate(BASES):
        for outcome, (a, p) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            operator = (
                a * np.kron(PAULI[basis[0]], PAULI["I"]) / 12
                + p * np.kron(PAULI["I"], PAULI[basis[1]]) / 12
                + a * p * np.kron(PAULI[basis[0]], PAULI[basis[1]]) / 4
            )
            weights[index, outcome] = operator[row, column]
    return weights


def _count_summary(counts: Any) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=float)
    rho = density_from_counts(counts)
    shots = counts.sum(axis=1)
    probabilities = counts / shots[:, None]
    zz = probabilities[BASES.index("ZZ")]
    coherences = {}
    for name, row, column, outcomes in (
        ("active_given_passive_0", 2, 0, [0, 2]),
        ("active_given_passive_1", 3, 1, [1, 3]),
        ("passive_given_active_0", 1, 0, [0, 1]),
        ("passive_given_active_1", 3, 2, [2, 3]),
        ("odd_sector_10_01", 2, 1, [1, 2]),
    ):
        weight = _coherence_weights(row, column)
        errors = []
        for w in (weight.real, weight.imag):
            variance = np.sum(
                (
                    np.sum(probabilities * w**2, axis=1)
                    - np.sum(probabilities * w, axis=1) ** 2
                )
                / shots
            )
            errors.append(float(np.sqrt(max(0.0, variance))))
        coherences[name] = dict(
            density_indices=[row, column],
            real=float(rho[row, column].real),
            imag=float(rho[row, column].imag),
            real_shot_se=errors[0],
            imag_shot_se=errors[1],
            conditioning_population=float(zz[outcomes].sum()),
            convention="Unnormalized joint density element rho[row,column]; no division or postselection",
        )
    return dict(
        raw_probabilities=zz.tolist(),
        zz_shots=int(shots[BASES.index("ZZ")]),
        rho_real=rho.real.tolist(),
        rho_imag=rho.imag.tolist(),
        conditional_coherences=coherences,
        odd_population=float(zz[1] + zz[2]),
        scope="Raw linear tomography, no positivity projection; binary outcomes do not resolve leakage",
    )


def _comparisons(summaries: Any) -> dict[str, Any]:
    result = {}
    for left, right in (
        ("parity", "standard"),
        ("standard", "idle"),
        ("parity", "idle"),
    ):
        comparison = {}
        for label, indices in (("P01", [1]), ("P10", [2]), ("Podd", [1, 2])):
            a, b = summaries[left], summaries[right]
            pa = float(np.asarray(a["raw_probabilities"])[indices].sum())
            pb = float(np.asarray(b["raw_probabilities"])[indices].sum())
            se = float(
                np.sqrt(pa * (1 - pa) / a["zz_shots"] + pb * (1 - pb) / b["zz_shots"])
            )
            comparison[label] = dict(
                difference=pa - pb,
                shot_se=se,
                ci95_shots=[pa - pb - 1.96 * se, pa - pb + 1.96 * se],
            )
        result[f"{left}_minus_{right}"] = comparison
    return result


def _check_identity(port: Any, expected: str) -> None:
    if _hash(_identity(port)) != expected:
        raise RuntimeError("Frozen diagnostic calibration identity changed")


def _validate_counts(row: Any, shots: int) -> NDArray[np.int64]:
    values = np.asarray(row["counts"])
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
        or values.sum() != shots
        or row["shots"] != shots
    ):
        raise ValueError("Diagnostic acquisition returned malformed counts")
    return np.asarray(values, dtype=np.int64)


def acquire_bswap_odd_sector_diagnostic(
    measurements: Any,
    directory: str | Path,
    *,
    shots: int = 1024,
    replicates: int = 2,
    seed: int = 20531,
    max_total_shots: int = 55_296,
) -> dict[str, Any]:
    """
    Compare standard/parity-cycled full-bSWAP x4 and matched idle from input 01.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Caller-owned connected session with frozen full recipe and current DRAG.
    directory : str or Path
        Fresh run-local directory; an existing protocol is never overwritten.
    shots, replicates : int, optional
        Positive counts per setting and independent repeated nine-basis blocks.
    seed : int, optional
        Randomization seed, not a device or calibration seed.
    max_total_shots : int, optional
        Maximum requested shots, including requests that subsequently fail.

    Returns
    -------
    dict
        `variant_summaries` maps standard/parity/idle to raw ZZ probabilities,
        linear density matrices and unnormalized conditional coherences. Intervals
        describe independent multinomial shot statistics only, not drift, SPAM,
        physical leakage, gate fidelity or a microscopic cause.

    Raises
    ------
    ValueError
        Budget, waveform equivalence, pulse headroom or count validation fails.
    RuntimeError
        Frozen calibration identity changes during acquisition.
    FileExistsError
        The destination already contains this diagnostic's protocol.

    Notes
    -----
    This function acquires hardware shots through the supplied port and writes
    raw IQ/count provenance, preflight waveforms, partial results and a summary.
    It does not connect, configure, recalibrate, adopt parameters or clear a
    failed admission check. The caller must enforce failure-only invocation.
    Actual emitted GE samples must match across all three variants; main and
    cancel tones must flip sign only on occurrences two and four. IDLE has the
    standard terminal logical frame solely to match physical analysis axes.
    """
    for name, value in (
        ("shots", shots),
        ("replicates", replicates),
        ("max_total_shots", max_total_shots),
    ):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    total = 3 * len(BASES) * replicates * shots
    if total > max_total_shots:
        raise ValueError(
            f"Diagnostic requires {total} shots, exceeding {max_total_shots}"
        )
    directory = Path(directory)
    if (directory / "protocol.json").exists():
        raise FileExistsError("Choose a fresh diagnostic directory")
    directory.mkdir(parents=True, exist_ok=True)
    identity = _identity(measurements)
    identity_hash = _hash(identity)
    recipes = deepcopy(measurements.recipes)
    variants, preflight = _preflight(measurements, recipes, directory)
    rng = np.random.default_rng(seed)
    units = [(replicate, basis) for replicate in range(replicates) for basis in BASES]
    rng.shuffle(units)
    permutations = list(itertools.permutations(VARIANTS))
    orders = [permutations[index % len(permutations)] for index in range(len(units))]
    rng.shuffle(orders)
    plan: list[dict[str, Any]] = [
        dict(replicate=replicate, basis=basis, variants=order)
        for (replicate, basis), order in zip(units, orders, strict=True)
    ]
    protocol = dict(
        diagnostic_only=True,
        scientific_qualified=False,
        shots=shots,
        replicates=replicates,
        seed=seed,
        planned_shots=total,
        maximum_shots=max_total_shots,
        prepared=PREPARED,
        bases=BASES,
        variants=variants,
        acquisition_plan=plan,
        identity=identity,
        identity_sha256=identity_hash,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    save_json(directory / "protocol.json", protocol)
    counts = np.zeros((3, replicates, 9, 4), dtype=np.int64)
    rows, requested = [], 0
    try:
        for unit in plan:
            for name in unit["variants"]:
                _check_identity(measurements, identity_hash)
                requested += shots
                save_json(
                    directory / "progress.json",
                    dict(requested_shots=requested, completed_requests=len(rows)),
                )
                row = measurements.acquire(
                    variants[name],
                    directory / "shots",
                    f"rep_{unit['replicate']:02d}_{unit['basis']}_{name}",
                    prepared=PREPARED,
                    basis=unit["basis"],
                    shots=shots,
                    recipes=recipes,
                )
                rows.append(
                    dict(
                        variant=name,
                        replicate=unit["replicate"],
                        basis=unit["basis"],
                        acquisition=row,
                    )
                )
                save_json(directory / "acquisitions.json", rows)
                _check_identity(measurements, identity_hash)
                values = _validate_counts(row, shots)
                counts[
                    VARIANTS.index(name), unit["replicate"], BASES.index(unit["basis"])
                ] = values
                np.savez_compressed(directory / "partial_counts.npz", counts=counts)
    except Exception as error:
        save_json(
            directory / "failure.json",
            dict(
                status="failed",
                diagnostic_only=True,
                scientific_qualified=False,
                error_type=type(error).__name__,
                error=str(error),
                requested_shots=requested,
                completed_requests=len(rows),
            ),
        )
        raise
    summaries, density = {}, np.empty((3, replicates, 4, 4), dtype=complex)
    for index, name in enumerate(VARIANTS):
        summary = _count_summary(counts[index].sum(axis=0))
        summary["replicates"] = [_count_summary(block) for block in counts[index]]
        summaries[name] = summary
        density[index] = [density_from_counts(block) for block in counts[index]]
    np.savez_compressed(
        directory / "counts_and_density.npz",
        counts=counts,
        raw_linear_density=density,
        variants=VARIANTS,
        bases=BASES,
    )
    result = dict(
        status="diagnostic_complete",
        diagnostic_only=True,
        scientific_qualified=False,
        requested_shots=requested,
        variant_summaries=summaries,
        comparisons=_comparisons(summaries),
        preflight=preflight,
        summary_path=str(directory / "summary.json"),
        inference_scope="Shot-statistical diagnostic only; no admission override, gate adoption or microscopic attribution",
    )
    save_json(directory / "summary.json", result)
    return result
