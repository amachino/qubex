"""
Paired-readout acquisition with initial-gg selection and an all-shot companion.

The caller owns hardware authorization, the Experiment connection, and classifier
validation. Selection is offline and is not active reset or a purity certificate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from qxpulse import Blank, PulseSchedule

from qubex.measurement.models.measurement_schedule import MeasurementSchedule

from .measurements import CampaignMeasurements, save_json
from .pulses import compile_campaign


def _predict_binary(classifier: Any, iq: np.ndarray, shots: int) -> np.ndarray:
    """Require one binary label per shot without truncating fractional labels."""
    labels = np.asarray(classifier.predict(iq))
    if labels.shape != (shots,) or not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Classifier must return one binary label per shot; raw IQ retained"
        )
    return labels.astype(np.int64)


def build_heralded_schedule(
    measurements: CampaignMeasurements,
    gates: Sequence[Any],
    *,
    settle_ns: float,
    prepared: Sequence[str] = ("0", "0"),
    basis: str = "ZZ",
    delay_ns: float = 0.0,
    recipes: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[MeasurementSchedule, dict[str, Any]]:
    """
    Build initial readout, settling idle, compiled controls, and final readout.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Caller-owned session with fixed target references and fresh DRAG pulses.
    gates : Sequence
        Logical gates accepted by `compile_campaign`.
    settle_ns : float
        Extra idle after the complete initial readout waveform, including its
        post-margin, and before the compiled preparation block. Must be finite,
        nonnegative, and on the native sample grid. No physically safe default
        is assumed; the caller must qualify photon depletion and backaction.
    prepared : Sequence[str], optional
        Active/passive preparation after the initial readout, default 00.
    basis : str, optional
        Final two-qubit analysis basis, default ZZ.
    delay_ns : float, optional
        Additional delay inside the compiled control block, in ns.
    recipes : Mapping or None, optional
        Explicit recipe snapshot; None uses the session recipes.

    Returns
    -------
    tuple[MeasurementSchedule, dict]
        Built schedule and physical capture/logical compilation timing metadata.
        The backend workaround capture is marked separately from initial/final.

    Raises
    ------
    ValueError
        Invalid settling time, stale backend prefix, or non-paired captures.

    Notes
    -----
    This function does not connect or acquire. Both readouts use the session's
    current readout pulse settings. The builder adds backend padding once. The
    herald-plus-settle prefix enters `global_start_ns`; backend padding enters
    `backend_preamble_ns` separately. The backend's final-readout guard is kept.
    Hardware capture spacing is not a resonator-ringdown qualification.
    """
    exp = measurements.exp
    profile = exp.ctx.measurement.constraint_profile
    sample = profile.sampling_period_ns
    settle = float(settle_ns)
    if (
        not math.isfinite(settle)
        or settle < 0
        or not np.isclose(settle / sample, round(settle / sample), atol=1e-9, rtol=0)
    ):
        raise ValueError("settle_ns must be finite, nonnegative, and sample aligned")
    backend_prefix = (
        profile.extra_capture_duration_ns if profile.require_workaround_capture else 0.0
    )
    if not np.isclose(
        measurements.backend_preamble_ns, backend_prefix, atol=1e-9, rtol=0
    ):
        raise ValueError(
            "Session backend prefix no longer matches its constraint profile"
        )
    registry = exp.ctx.experiment_system.target_registry
    read_labels = {
        q: registry.resolve_read_label(q, allow_legacy=True)
        for q in measurements.qubits
    }
    labels = [
        *measurements.qubits,
        measurements.drive_label,
        measurements.cancel_label,
        *read_labels.values(),
    ]
    readouts = {q: exp.pulse.readout(q) for q in measurements.qubits}
    selected = measurements.recipes if recipes is None else recipes
    with PulseSchedule(labels) as parent:
        for q, label in read_labels.items():
            parent.add(label, deepcopy(readouts[q]))
        parent.barrier()
        readout_end = float(parent.duration)
        parent.add(measurements.qubits[0], Blank(settle, sampling_period=sample))
        parent.barrier()
        prefix = float(parent.duration)
        child, compiled = compile_campaign(
            gates,
            recipes=selected,
            qubits=measurements.qubits,
            drive_label=measurements.drive_label,
            cancel_label=measurements.cancel_label,
            target_frequencies_ghz=measurements.targets,
            reference_frequencies_ghz=measurements.references,
            rabi_ghz_per_amplitude=measurements.rabi_scale,
            x90=measurements.x90,
            xpi=measurements.xpi,
            global_start_ns=prefix,
            backend_preamble_ns=backend_prefix,
            prepared=prepared,
            basis=basis,
            delay_ns=delay_ns,
        )
        parent.call(child)
        parent.barrier()
        guard = profile.final_readout_guard_duration_ns
        parent.add(measurements.qubits[0], Blank(guard, sampling_period=sample))
        parent.barrier()
        quantum = profile.word_duration_ns or sample
        alignment = (-float(parent.duration)) % quantum
        if alignment:
            parent.add(measurements.qubits[0], Blank(alignment, sampling_period=sample))
            parent.barrier()
        for q, label in read_labels.items():
            parent.add(label, deepcopy(readouts[q]))
        parent.barrier()
    built = exp.build_measurement_schedule(
        parent,
        frequencies=measurements.targets,
        final_measurement=False,
        capture_placement="pulse_aligned",
        plot=False,
    )
    capture_metadata = {}
    for q, label in read_labels.items():
        captures = sorted(
            built.capture_schedule.channels[label], key=lambda c: c.start_time
        )
        real = [capture for capture in captures if not capture.is_workaround]
        if len(real) != 2:
            raise ValueError(f"Expected two pulse-aligned readout captures for {q}")
        capture_metadata[q] = [
            dict(
                capture_index=index,
                role=("initial", "final")[index],
                start_ns=float(capture.start_time),
                duration_ns=float(capture.duration),
            )
            for index, capture in enumerate(real)
        ]
    return built, dict(
        compiled=compiled,
        capture_order=["initial", "final"],
        captures=capture_metadata,
        read_labels=read_labels,
        initial_readout_waveform_end_ns=readout_end,
        settle_ns=settle,
        herald_prefix_ns=prefix,
        final_readout_guard_ns=guard,
        backend_preamble_ns=backend_prefix,
        physical_sequence_duration_ns=float(built.pulse_schedule.duration),
        target_frequencies_ghz=dict(measurements.targets),
        reference_frequencies_ghz=dict(measurements.references),
    )


async def acquire_heralded(
    measurements: CampaignMeasurements,
    gates: Sequence[Any],
    directory: str | Path,
    label: str,
    *,
    settle_ns: float,
    prepared: Sequence[str] = ("0", "0"),
    basis: str = "ZZ",
    delay_ns: float = 0.0,
    shots: int | None = None,
    recipes: Mapping[str, Mapping[str, Any]] | None = None,
    classifier_source: str | None = None,
) -> dict[str, Any]:
    """
    Save paired per-shot IQ and classify all shots plus initial-gg-selected shots.

    Parameters
    ----------
    measurements : CampaignMeasurements
        Caller-owned authorized session. Classifiers are copied for this call.
    gates : Sequence
        Logical gates accepted by `compile_campaign`.
    directory : str or Path
        New raw IQ, classification, and JSON files are saved here.
    label : str
        Acquisition filename label without directory components.
    settle_ns : float
        Extra idle after initial readout, in ns; see `build_heralded_schedule`.
    prepared : Sequence[str], optional
        Active/passive preparation after initial readout, default 00.
    basis : str, optional
        Final two-qubit analysis basis, default ZZ.
    delay_ns : float, optional
        Additional pre-preparation delay, in ns.
    shots : int or None, optional
        Positive integer requested shots; None uses the session default.
    recipes : Mapping or None, optional
        Explicit recipe snapshot; None uses the session recipes.
    classifier_source : str or None, optional
        Path or identifier for the separately archived classifier calibration.

    Returns
    -------
    dict
        Both count vectors, acceptance, file paths, and schedule provenance.
        Conditional probabilities are None when no initial-gg shots remain.
        No SPAM inversion, clipping, or final-outcome selection is applied.

    Raises
    ------
    TimeoutError
        Session deadline is within one minute before acquisition.
    ValueError
        Invalid inputs, capture count, per-shot IQ payload, or classifier labels.
        Acquired payloads are saved before capture or classification validation.

    Notes
    -----
    This function performs one real acquisition using the existing connection;
    it never connects, configures, adopts a recipe, or updates a classifier.
    Files use unique names and do not replace production data. Integrated IQ
    means one complex sample per capture per shot, not the full ADC trace.
    Capture order and shot indices are retained without sorting or truncation.
    Initial gg means classifier-assigned gg, not verified physical ground-state
    purity. An all-shot companion has the same initial readout and backaction;
    a separate duration-matched no-herald control is needed to test that effect.
    """
    deadline = measurements.deadline
    if deadline is not None and datetime.now(deadline.tzinfo) >= deadline - timedelta(
        seconds=60
    ):
        raise TimeoutError(
            "Reservation is ending; stop before another hardware request"
        )
    requested = measurements.shots if shots is None else shots
    if (
        isinstance(requested, (bool, np.bool_))
        or not isinstance(requested, (int, np.integer))
        or requested <= 0
    ):
        raise ValueError("shots must be a positive integer")
    requested = int(requested)
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError("label must be a filename label without directory components")
    selected = deepcopy(dict(measurements.recipes if recipes is None else recipes))
    classifiers = deepcopy(measurements.classifiers)
    schedule, schedule_record = build_heralded_schedule(
        measurements,
        gates,
        settle_ns=settle_ns,
        prepared=prepared,
        basis=basis,
        delay_ns=delay_ns,
        recipes=selected,
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{measurements.session_id}_{uuid4().hex}_{label}"
    iq_path = directory / f"{stem}_iq.npz"
    json_path = directory / f"{stem}.json"
    started_at = datetime.now().astimezone().isoformat()
    clock = perf_counter()
    result = await measurements.exp.run_measurement(
        schedule,
        n_shots=requested,
        shot_interval=measurements.shot_interval_ns,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
        final_measurement=False,
    )
    row: dict[str, Any] = dict(
        label=label,
        shots=requested,
        qubits=list(measurements.qubits),
        gates=gates,
        prepared=prepared,
        basis=basis,
        delay_ns=delay_ns,
        recipes=selected,
        classifier_source=classifier_source,
        iq_file=str(iq_path),
        acquisition_started_at=started_at,
        acquisition_ended_at=datetime.now().astimezone().isoformat(),
        acquisition_duration_seconds=perf_counter() - clock,
        shot_interval_idle_ns=measurements.shot_interval_ns,
        selection_rule="initial active label == 0 AND initial passive label == 0",
        selection_uses_final_outcome=False,
        physical_ground_state_purity_verified=False,
        status="iq_saved_unclassified",
        **schedule_record,
    )
    payload: dict[str, Any] = dict(
        qubits=np.asarray(measurements.qubits),
        shot_index=np.arange(requested),
        capture_order=np.asarray(["initial", "final"]),
    )
    captures_by_qubit = [result.data.get(q, []) for q in measurements.qubits]
    for qi, captures in enumerate(captures_by_qubit):
        for ci, capture in enumerate(captures):
            payload[f"q{qi}_capture{ci}"] = np.asarray(capture.data)
    paired_shape = all(
        len(captures) == 2
        and all(
            capture.iq_series is not None
            and np.asarray(capture.iq_series).shape == (requested,)
            for capture in captures
        )
        for captures in captures_by_qubit
    )
    if paired_shape:
        payload["iq_initial"] = np.stack(
            [np.asarray(c[0].iq_series) for c in captures_by_qubit]
        )
        payload["iq_final"] = np.stack(
            [np.asarray(c[1].iq_series) for c in captures_by_qubit]
        )
    np.savez_compressed(iq_path, **payload)
    row["returned_capture_counts"] = [len(captures) for captures in captures_by_qubit]
    save_json(json_path, row)
    capture_error = None
    if any(len(captures) != 2 for captures in captures_by_qubit):
        capture_error = "Expected exactly two captures per qubit; raw payloads retained"
    elif not paired_shape:
        capture_error = (
            "Expected per-shot iq_series of the requested length; raw payloads retained"
        )
    elif (
        not np.isfinite(payload["iq_initial"]).all()
        or not np.isfinite(payload["iq_final"]).all()
    ):
        capture_error = "Nonfinite paired IQ; raw payloads retained"
    if capture_error is not None:
        row.update(status="invalid_capture_data", error=capture_error)
        save_json(json_path, row)
        raise ValueError(capture_error)
    initial_iq, final_iq = payload["iq_initial"], payload["iq_final"]
    try:
        initial = np.stack(
            [
                _predict_binary(classifiers[q], initial_iq[qi], requested)
                for qi, q in enumerate(measurements.qubits)
            ]
        )
        final = np.stack(
            [
                _predict_binary(classifiers[q], final_iq[qi], requested)
                for qi, q in enumerate(measurements.qubits)
            ]
        )
    except Exception as exc:
        row.update(status="classification_failed", error=f"{type(exc).__name__}: {exc}")
        save_json(json_path, row)
        raise
    mask = np.all(initial == 0, axis=0)
    accepted = int(np.count_nonzero(mask))
    outcomes = 2 * final[0] + final[1]
    all_counts = np.bincount(outcomes, minlength=4)
    selected_counts = np.bincount(outcomes[mask], minlength=4)
    classification_path = directory / f"{stem}_classification.npz"
    np.savez_compressed(
        classification_path,
        initial_labels=initial,
        final_labels=final,
        initial_gg_mask=mask,
        counts_allshots=all_counts,
        counts_initial_gg=selected_counts,
    )
    row.update(
        status="classified" if accepted else "no_accepted_shots",
        counts_allshots=all_counts.tolist(),
        counts_initial_gg=selected_counts.tolist(),
        accepted_shots=accepted,
        acceptance_fraction=accepted / requested,
        raw_probabilities_allshots=(all_counts / requested).tolist(),
        raw_probabilities_initial_gg=(selected_counts / accepted).tolist()
        if accepted
        else None,
        classification_file=str(classification_path),
    )
    save_json(json_path, row)
    return row
