"""One-dimensional repetition-code experiment helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import Blank, PulseSchedule

Basis = Literal["bit", "phase"]


def repetition_code(
    exp: Experiment,
    qubits: Sequence[str],
    rounds: int,
    *,
    basis: Basis = "bit",
    initial: str | None = None,
    n_shots: int | None = None,
    measure_threshold: float | None = 0.7,
    data_threshold: float | None = 0.6,
    dynamical_decoupling: bool = True,
    readout_alignment_ns: float | None = 8.0,
    run_analysis: bool = True,
    return_raw_result: bool = False,
    plot_schedule: bool = False,
    plot_analysis: bool = False,
    measure_readout_duration: float | None = None,
    data_readout_duration: float | None = None,
    readout_amplitudes: Mapping[str, float] | None = None,
    readout_pre_margin: float | None = None,
    readout_post_margin: float | None = None,
    dd_pulses: int = 2,
    **execute_options: Any,
) -> Result:
    """
    Run a one-dimensional repetition-code experiment.

    Parameters
    ----------
    exp
        Experiment instance used to build pulses and execute the schedule.
    qubits
        Physical chain ordered as data, measure, data, measure, ..., data.
    rounds
        Number of syndrome-extraction rounds.
    basis
        ``"bit"`` for Z-basis bit-flip detection or ``"phase"`` for X-basis
        phase-flip detection. Phase-basis records are converted to Z readout
        records by Hadamards around the CNOT body and before final readout.
    initial
        Logical initial state. Defaults to ``"0"`` for ``basis="bit"`` and
        ``"+"`` for ``basis="phase"``.
    n_shots
        Number of shots passed to :meth:`Experiment.execute`.
    measure_threshold, data_threshold
        Optional classification confidence thresholds. Labels outside
        ``{0, 1}`` are dropped from the valid-shot set.
    dynamical_decoupling
        If True, insert CPMG-like X pulses on data qubits during intermediate
        syndrome-readout windows.
    readout_alignment_ns
        Required capture-start alignment in ns. Use ``None`` to disable padding
        and validation.
    run_analysis
        If True, compute detector records, logical-error metrics, hit rates, and
        detector two-point correlations.
    return_raw_result
        If True, include the raw execute result under ``raw_result``.
    plot_schedule
        If True, plot the constructed schedule before execution.
    plot_analysis
        If True, display a logical-error summary plot and detector-correlation
        heatmap using the returned analysis payload.
    measure_readout_duration, data_readout_duration
        Optional readout-duration overrides for syndrome and final data readout.
    readout_amplitudes
        Optional readout-amplitude overrides keyed by qubit or readout label.
    readout_pre_margin, readout_post_margin
        Optional readout margin overrides.
    dd_pulses
        Number of equally centered data-qubit X pulses during readout windows.
    **execute_options
        Additional options forwarded to :meth:`Experiment.execute`.

    Returns
    -------
    Result
        Result payload containing valid shots, detector/correlation data, and
        logical-error metrics.
    """
    layout = _build_layout(exp, qubits)
    rounds = _validate_rounds(rounds)
    basis = _normalize_basis(basis)
    initial = _normalize_initial(initial, basis=basis)

    readout_amplitudes_by_label = _normalize_readout_amplitudes(
        exp,
        readout_amplitudes,
    )
    schedule = _build_repetition_schedule(
        exp,
        layout=layout,
        rounds=rounds,
        basis=basis,
        initial=initial,
        dynamical_decoupling=dynamical_decoupling,
        readout_alignment_ns=readout_alignment_ns,
        measure_readout_duration=measure_readout_duration,
        data_readout_duration=data_readout_duration,
        readout_amplitudes=readout_amplitudes_by_label,
        readout_pre_margin=readout_pre_margin,
        readout_post_margin=readout_post_margin,
        dd_pulses=dd_pulses,
    )
    alignment_table = _assert_readout_alignment(
        schedule,
        readout_labels=[
            layout.readout_labels[q] for q in layout.measure_qubits + layout.data_qubits
        ],
        word_ns=readout_alignment_ns,
    )

    if plot_schedule:
        schedule.plot()

    execute_kwargs = dict(execute_options)
    execute_kwargs.setdefault("mode", "single")
    execute_kwargs.setdefault("final_measurement", False)
    if n_shots is not None:
        execute_kwargs.setdefault("n_shots", int(n_shots))

    raw_result = exp.execute(schedule, **execute_kwargs)
    valid_shots, record_array, requested_shots = _extract_valid_shots(
        raw_result,
        data_qubits=layout.data_qubits,
        measure_qubits=layout.measure_qubits,
        rounds=rounds,
        measure_threshold=measure_threshold,
        data_threshold=data_threshold,
    )

    payload: dict[str, object] = {
        "qubits": list(layout.qubits),
        "data_qubits": list(layout.data_qubits),
        "measure_qubits": list(layout.measure_qubits),
        "readout_labels": dict(layout.readout_labels),
        "cnot_pairs": list(layout.cnot_pairs),
        "cr_labels": list(layout.cr_labels),
        "distance": layout.distance,
        "rounds": rounds,
        "basis": basis,
        "initial": initial,
        "valid_shots": valid_shots,
        "valid_fraction": len(valid_shots) / requested_shots
        if requested_shots
        else np.nan,
        "shots_requested": requested_shots,
        "shots_valid": len(valid_shots),
        "record_array": record_array,
        "alignment_table": alignment_table,
        "schedule_duration_ns": float(schedule.duration),
    }

    if run_analysis:
        analysis_payload = _analyze_valid_records(
            valid_shots,
            record_array=record_array,
            distance=layout.distance,
            rounds=rounds,
            initial=initial,
            basis=basis,
        )
        payload.update(analysis_payload)

    if return_raw_result:
        payload["raw_result"] = raw_result

    result = Result(data=payload)
    if plot_analysis:
        plot_repetition_code_analysis(result)
    return result


class _Layout:
    def __init__(
        self,
        *,
        qubits: list[str],
        data_qubits: list[str],
        measure_qubits: list[str],
        readout_labels: dict[str, str],
        cnot_pairs: list[tuple[str, str]],
        cr_labels: list[str],
    ) -> None:
        self.qubits = qubits
        self.data_qubits = data_qubits
        self.measure_qubits = measure_qubits
        self.readout_labels = readout_labels
        self.cnot_pairs = cnot_pairs
        self.cr_labels = cr_labels
        self.distance = len(data_qubits)


def _build_layout(exp: Experiment, qubits: Sequence[str]) -> _Layout:
    qubit_list = [str(q) for q in qubits]
    if len(qubit_list) < 5 or len(qubit_list) % 2 == 0:
        raise ValueError(
            "qubits must be an odd-length physical chain with at least 5 entries "
            "(data, measure, data, ..., data)."
        )

    data_qubits = qubit_list[0::2]
    measure_qubits = qubit_list[1::2]
    if len(data_qubits) != len(measure_qubits) + 1:
        raise ValueError(
            "Invalid repetition-code chain: expected one more data qubit than measure qubits."
        )

    cnot_pairs: list[tuple[str, str]] = []
    for stabilizer_index, measure_qubit in enumerate(measure_qubits):
        cnot_pairs.append((data_qubits[stabilizer_index], measure_qubit))
        cnot_pairs.append((data_qubits[stabilizer_index + 1], measure_qubit))

    readout_labels = {qubit: _resolve_readout_label(exp, qubit) for qubit in qubit_list}
    cr_labels = [
        _resolve_cnot_channel_label(exp, control, target)
        for control, target in cnot_pairs
    ]
    return _Layout(
        qubits=qubit_list,
        data_qubits=data_qubits,
        measure_qubits=measure_qubits,
        readout_labels=readout_labels,
        cnot_pairs=cnot_pairs,
        cr_labels=cr_labels,
    )


def _validate_rounds(rounds: int) -> int:
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")
    return rounds


def _normalize_basis(basis: str) -> Basis:
    normalized = str(basis).strip().lower().replace("_", "-")
    if normalized in {"bit", "z", "bit-flip", "bitflip"}:
        return "bit"
    if normalized in {"phase", "x", "phase-flip", "phaseflip"}:
        return "phase"
    raise ValueError("basis must be one of 'bit' or 'phase'.")


def _normalize_initial(initial: str | None, *, basis: Basis) -> str:
    if initial is None:
        return "0" if basis == "bit" else "+"
    normalized = str(initial).strip().lower().replace(" ", "")
    if normalized == "plus":
        normalized = "+"
    if normalized == "minus":
        normalized = "-"
    allowed = {"0", "1"} if basis == "bit" else {"+", "-"}
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(repr(item) for item in allowed))
        raise ValueError(f"initial must be one of {allowed_text} for basis={basis!r}.")
    return normalized


def _logical_reference_bit(initial: str, *, basis: Basis) -> int:
    if basis == "phase":
        return 0 if initial == "+" else 1
    return int(initial)


def _resolve_readout_label(exp: Experiment, qubit: str) -> str:
    resolver = getattr(getattr(exp, "ctx", None), "resolve_read_label", None)
    if callable(resolver):
        return str(resolver(qubit))
    return "R" + qubit


def _resolve_cnot_channel_label(exp: Experiment, control: str, target: str) -> str:
    direct_label = f"{control}-{target}"
    reverse_label = f"{target}-{control}"
    calib_note = getattr(getattr(exp, "ctx", None), "calib_note", None)
    cr_params = getattr(calib_note, "cr_params", None)
    if cr_params is None and calib_note is not None:
        getter = getattr(calib_note, "get", None)
        if not callable(getter):
            getter = None
    else:
        getter = None
    if cr_params is None and getter is not None:
        try:
            cr_params = getter("cr_params")
        except Exception:
            cr_params = None
    if isinstance(cr_params, Mapping):
        if direct_label in cr_params:
            return direct_label
        if reverse_label in cr_params:
            return reverse_label
    return direct_label


def _normalize_readout_amplitudes(
    exp: Experiment,
    readout_amplitudes: Mapping[str, float] | None,
) -> dict[str, float]:
    if readout_amplitudes is None:
        return {}
    normalized: dict[str, float] = {}
    for label, amplitude in readout_amplitudes.items():
        label_text = str(label)
        normalized[label_text] = float(amplitude)
        with suppress(Exception):
            normalized[_resolve_readout_label(exp, label_text)] = float(amplitude)
    return normalized


def _build_repetition_schedule(
    exp: Experiment,
    *,
    layout: _Layout,
    rounds: int,
    basis: Basis,
    initial: str,
    dynamical_decoupling: bool,
    readout_alignment_ns: float | None,
    measure_readout_duration: float | None,
    data_readout_duration: float | None,
    readout_amplitudes: Mapping[str, float],
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
    dd_pulses: int,
) -> PulseSchedule:
    labels = _schedule_labels(layout)
    anchor = layout.data_qubits[0]
    with PulseSchedule(labels) as schedule:
        _prepare_initial_state(schedule, exp, layout.data_qubits, initial)

        for round_index in range(rounds):
            _append_repetition_round(schedule, exp, layout, basis=basis)
            schedule.barrier()
            _pad_to_readout_alignment(
                schedule,
                exp,
                anchor_label=anchor,
                readout_qubits=layout.measure_qubits,
                readout_duration=measure_readout_duration,
                readout_amplitudes=readout_amplitudes,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                word_ns=readout_alignment_ns,
            )
            _append_readouts(
                schedule,
                exp,
                layout,
                qubits=layout.measure_qubits,
                readout_duration=measure_readout_duration,
                readout_amplitudes=readout_amplitudes,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
            )
            if round_index != rounds - 1 and dynamical_decoupling:
                _append_readout_window_dd(
                    schedule,
                    exp,
                    data_qubits=layout.data_qubits,
                    readout_duration=measure_readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    dd_pulses=dd_pulses,
                )
            schedule.barrier()

        if basis == "phase":
            _append_hadamards(schedule, exp, layout.data_qubits)
            schedule.barrier()

        _pad_to_readout_alignment(
            schedule,
            exp,
            anchor_label=anchor,
            readout_qubits=layout.data_qubits,
            readout_duration=data_readout_duration,
            readout_amplitudes=readout_amplitudes,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            word_ns=readout_alignment_ns,
        )
        _append_readouts(
            schedule,
            exp,
            layout,
            qubits=layout.data_qubits,
            readout_duration=data_readout_duration,
            readout_amplitudes=readout_amplitudes,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
        )
        schedule.barrier()
    return schedule


def _schedule_labels(layout: _Layout) -> list[str]:
    labels = list(layout.qubits)
    for cr_label in layout.cr_labels:
        if cr_label not in labels:
            labels.append(cr_label)
    for label in layout.readout_labels.values():
        if label not in labels:
            labels.append(label)
    return labels


def _prepare_initial_state(
    schedule: PulseSchedule,
    exp: Experiment,
    data_qubits: Sequence[str],
    initial: str,
) -> None:
    schedule.barrier()
    for qubit in data_qubits:
        if initial == "1":
            schedule.add(qubit, exp.x180(qubit))
        elif initial == "+":
            schedule.add(qubit, exp.hadamard(qubit))
        elif initial == "-":
            schedule.add(qubit, exp.hadamard(qubit))
            schedule.add(qubit, exp.z180())
    schedule.barrier()


def _append_repetition_round(
    schedule: PulseSchedule,
    exp: Experiment,
    layout: _Layout,
    *,
    basis: Basis,
) -> None:
    if basis == "phase":
        _append_hadamards(schedule, exp, layout.data_qubits)
        schedule.barrier()

    for control, target in layout.cnot_pairs:
        schedule.call(exp.cnot(control, target))

    if basis == "phase":
        schedule.barrier()
        _append_hadamards(schedule, exp, layout.data_qubits)


def _append_hadamards(
    schedule: PulseSchedule,
    exp: Experiment,
    qubits: Sequence[str],
) -> None:
    for qubit in qubits:
        schedule.add(qubit, exp.hadamard(qubit))


def _append_readouts(
    schedule: PulseSchedule,
    exp: Experiment,
    layout: _Layout,
    *,
    qubits: Sequence[str],
    readout_duration: float | None,
    readout_amplitudes: Mapping[str, float],
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
) -> None:
    for qubit in qubits:
        label = layout.readout_labels[qubit]
        schedule.add(
            label,
            exp.readout(
                qubit,
                duration=readout_duration,
                amplitude=readout_amplitudes.get(label, readout_amplitudes.get(qubit)),
                pre_margin=readout_pre_margin,
                post_margin=readout_post_margin,
            ),
        )


def _append_readout_window_dd(
    schedule: PulseSchedule,
    exp: Experiment,
    *,
    data_qubits: Sequence[str],
    readout_duration: float | None,
    readout_pre_margin: float | None,
    dd_pulses: int,
) -> None:
    dd_pulses = int(dd_pulses)
    if dd_pulses <= 0:
        return

    duration = float(
        readout_duration if readout_duration is not None else exp.pulse.readout_duration
    )
    pre_margin = float(
        readout_pre_margin
        if readout_pre_margin is not None
        else exp.pulse.readout_pre_margin
    )
    for qubit in data_qubits:
        if pre_margin > 0:
            schedule.add(qubit, Blank(pre_margin))
        pi_pulse = exp.x180(qubit)
        pulse_duration = float(pi_pulse.duration)
        elapsed = 0.0
        for pulse_index in range(dd_pulses):
            center = duration * (2 * pulse_index + 1) / (2 * dd_pulses)
            blank = center - elapsed - pulse_duration / 2
            if blank < -1e-9:
                raise ValueError(
                    f"DD pulses do not fit in the readout window for {qubit}: "
                    f"duration={duration}, pulse_duration={pulse_duration}, dd_pulses={dd_pulses}."
                )
            if blank > 0:
                schedule.add(qubit, Blank(blank))
            schedule.add(qubit, pi_pulse)
            elapsed = center + pulse_duration / 2
        trailing = duration - elapsed
        if trailing > 0:
            schedule.add(qubit, Blank(trailing))


def _readout_internal_start_offset_ns(
    exp: Experiment,
    *,
    qubit: str,
    readout_label: str,
    readout_duration: float | None,
    readout_amplitude: float | None,
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
) -> float:
    with PulseSchedule([readout_label]) as tmp:
        tmp.add(
            readout_label,
            exp.readout(
                qubit,
                duration=readout_duration,
                amplitude=readout_amplitude,
                pre_margin=readout_pre_margin,
                post_margin=readout_post_margin,
            ),
        )
    ranges = tmp.get_pulse_ranges([readout_label]).get(readout_label, [])
    if not ranges:
        return 0.0
    return float(ranges[0].start) * _schedule_sampling_period(tmp, readout_label)


def _pad_to_readout_alignment(
    schedule: PulseSchedule,
    exp: Experiment,
    *,
    anchor_label: str,
    readout_qubits: Sequence[str],
    readout_duration: float | None,
    readout_amplitudes: Mapping[str, float],
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
    word_ns: float | None,
) -> float:
    if word_ns is None:
        return 0.0
    word_ns = float(word_ns)
    if word_ns <= 0:
        raise ValueError("readout_alignment_ns must be positive or None.")
    if not readout_qubits:
        return 0.0

    offsets = []
    for qubit in readout_qubits:
        readout_label = _resolve_readout_label(exp, qubit)
        offsets.append(
            _readout_internal_start_offset_ns(
                exp,
                qubit=qubit,
                readout_label=readout_label,
                readout_duration=readout_duration,
                readout_amplitude=readout_amplitudes.get(
                    readout_label, readout_amplitudes.get(qubit)
                ),
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
            )
        )
    remainders = {round(offset % word_ns, 9) for offset in offsets}
    if len(remainders) > 1:
        raise RuntimeError(
            "Readout internal offsets are not mutually aligned: "
            f"{dict(zip(readout_qubits, offsets, strict=True))}"
        )

    effective_start_ns = float(schedule.duration) + offsets[0]
    pad_ns = (word_ns - (effective_start_ns % word_ns)) % word_ns
    if pad_ns < 1e-9 or abs(pad_ns - word_ns) < 1e-9:
        pad_ns = 0.0
    if pad_ns > 0:
        schedule.add(anchor_label, Blank(pad_ns))
        schedule.barrier()
    return float(pad_ns)


def _assert_readout_alignment(
    schedule: PulseSchedule,
    *,
    readout_labels: Sequence[str],
    word_ns: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if word_ns is None:
        return rows

    word_ns = float(word_ns)
    ranges_by_label = schedule.get_pulse_ranges(list(readout_labels))
    bad: list[tuple[str, int, float, float]] = []
    for label in readout_labels:
        sampling_period = _schedule_sampling_period(schedule, label)
        for index, pulse_range in enumerate(ranges_by_label.get(label, [])):
            start_ns = float(pulse_range.start) * sampling_period
            duration_ns = float(len(pulse_range)) * sampling_period
            start_aligned = _is_multiple(start_ns, word_ns)
            duration_aligned = _is_multiple(duration_ns, word_ns)
            aligned = start_aligned and duration_aligned
            rows.append(
                {
                    "label": label,
                    "index": index,
                    "start_ns": start_ns,
                    "duration_ns": duration_ns,
                    "aligned": aligned,
                }
            )
            if not aligned:
                bad.append((label, index, start_ns, duration_ns))
    if bad:
        raise RuntimeError(f"Readout capture is not {word_ns:g} ns aligned: {bad}")
    return rows


def _schedule_sampling_period(schedule: PulseSchedule, label: str) -> float:
    sequence = schedule.get_sequence(label, copy=False)
    return float(sequence.sampling_period)


def _is_multiple(value: float, step: float) -> bool:
    ratio = value / step
    return abs(ratio - round(ratio)) <= 1e-9


def _extract_valid_shots(
    result: Any,
    *,
    data_qubits: Sequence[str],
    measure_qubits: Sequence[str],
    rounds: int,
    measure_threshold: float | None,
    data_threshold: float | None,
) -> tuple[list[dict[str, list[Any]]], NDArray[np.int8], int]:
    measure_labels = {
        (qubit, round_index): _classified_capture(
            result.data[qubit][round_index],
            threshold=measure_threshold,
        )
        for round_index in range(rounds)
        for qubit in measure_qubits
    }
    data_labels = {
        qubit: _classified_capture(result.data[qubit][-1], threshold=data_threshold)
        for qubit in data_qubits
    }

    shot_counts = [len(values) for values in measure_labels.values()] + [
        len(values) for values in data_labels.values()
    ]
    if not shot_counts:
        raise RuntimeError("No measurement captures were found in the execute result.")
    requested_shots = min(shot_counts)

    valid_shots: list[dict[str, list[Any]]] = []
    record_rows: list[list[int]] = []
    for shot_index in range(requested_shots):
        shot_meas: list[list[int]] = []
        record_row: list[int] = []
        valid = True
        for round_index in range(rounds):
            round_bits: list[int] = []
            for qubit in measure_qubits:
                bit = _binary_label(measure_labels[(qubit, round_index)][shot_index])
                if bit is None:
                    valid = False
                    break
                round_bits.append(bit)
                record_row.append(bit)
            if not valid:
                break
            shot_meas.append(round_bits)
        if not valid:
            continue

        data_bits: list[int] = []
        for qubit in data_qubits:
            bit = _binary_label(data_labels[qubit][shot_index])
            if bit is None:
                valid = False
                break
            data_bits.append(bit)
            record_row.append(bit)
        if not valid:
            continue

        valid_shots.append({"meas": shot_meas, "data": data_bits})
        record_rows.append(record_row)

    return valid_shots, np.asarray(record_rows, dtype=np.int8), int(requested_shots)


def _classified_capture(capture: Any, *, threshold: float | None) -> NDArray[np.int_]:
    try:
        return np.asarray(capture.get_classified_data(threshold=threshold), dtype=int)
    except Exception:
        classified = getattr(capture, "classified", None)
        if classified is None:
            raise
        return np.asarray(classified, dtype=int)


def _binary_label(value: Any) -> int | None:
    try:
        bit = int(value)
    except Exception:
        return None
    if bit in (0, 1):
        return bit
    return None


def _analyze_valid_records(
    valid_shots: Sequence[dict[str, list[Any]]],
    *,
    record_array: NDArray[np.int8],
    distance: int,
    rounds: int,
    initial: str,
    basis: Basis,
) -> dict[str, object]:
    if len(valid_shots) == 0:
        empty_detectors = np.empty((0, rounds + 1, distance - 1), dtype=np.int8)
        return {
            "detector_array": empty_detectors,
            "detector_labels": _detector_labels(distance, rounds),
            "logical_error_rate": np.nan,
            "logical_error_rate_per_round_linear": np.nan,
            "logical_error_rate_per_round_iid": np.nan,
            "num_logical_errors": 0,
            "detector_hit_rates": [],
            "correlation_matrix": np.empty((0, 0), dtype=float),
        }

    detector_array = np.asarray(
        [_detectors_from_shot(shot["meas"], shot["data"]) for shot in valid_shots],
        dtype=np.int8,
    )
    detector_flat = detector_array.reshape(detector_array.shape[0], -1)
    data_bits = np.asarray([shot["data"] for shot in valid_shots], dtype=np.int8)
    reference = _logical_reference_bit(initial, basis=basis)
    measured_observable = np.bitwise_xor.reduce(data_bits ^ reference, axis=1)
    predicted_observable = _decode_observables(
        detector_flat, distance=distance, rounds=rounds
    )
    logical_errors = predicted_observable != measured_observable
    num_logical_errors = int(np.count_nonzero(logical_errors))
    logical_error_rate = float(num_logical_errors / len(valid_shots))

    return {
        "record_array": record_array,
        "detector_array": detector_array,
        "detector_labels": _detector_labels(distance, rounds),
        "predicted_observable": predicted_observable,
        "measured_observable": measured_observable,
        "logical_errors": logical_errors,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_per_round_linear": float(logical_error_rate / rounds),
        "logical_error_rate_per_round_iid": float(
            1.0 - (1.0 - logical_error_rate) ** (1.0 / rounds)
        ),
        "num_logical_errors": num_logical_errors,
        "detector_hit_rates": _detector_hit_rates(detector_array),
        "correlation_matrix": _two_point_correlation(detector_flat),
    }


def _detectors_from_shot(
    measure_records: Sequence[Sequence[int]],
    data_bits: Sequence[int],
) -> list[list[int]]:
    if not measure_records:
        raise ValueError("measure_records must not be empty.")
    rows = [[0 for _ in measure_records[0]], [0 for _ in measure_records[0]]]
    rows.extend([list(map(int, row)) for row in measure_records])
    final_parity = [
        int(data_bits[i]) ^ int(data_bits[i + 1]) ^ rows[-1][i]
        for i in range(len(data_bits) - 1)
    ]
    rows.append(final_parity)
    return [
        [
            int(rows[row_index][col] ^ rows[row_index + 2][col])
            for col in range(len(measure_records[0]))
        ]
        for row_index in range(len(rows) - 2)
    ]


def _detector_labels(distance: int, rounds: int) -> list[str]:
    labels: list[str] = []
    for detector_row in range(rounds + 1):
        if detector_row == 0:
            row_label = "round0_boundary"
        elif detector_row == 1 and rounds > 1:
            row_label = "round1_boundary"
        elif detector_row < rounds:
            row_label = f"round{detector_row}_time"
        else:
            row_label = "final_data_boundary"
        labels.extend(
            f"{row_label}:S{stabilizer}" for stabilizer in range(distance - 1)
        )
    return labels


def _detector_hit_rates(detector_array: NDArray[np.int8]) -> list[dict[str, object]]:
    if detector_array.size == 0:
        return []
    rates = detector_array.mean(axis=0)
    rows: list[dict[str, object]] = []
    rounds_plus_final, stabilizers = rates.shape
    for detector_row in range(rounds_plus_final):
        if detector_row == 0:
            row_label = "round0_boundary"
        elif detector_row == 1 and rounds_plus_final > 2:
            row_label = "round1_boundary"
        elif detector_row < rounds_plus_final - 1:
            row_label = f"round{detector_row}_time"
        else:
            row_label = "final_data_boundary"
        rows.extend(
            {
                "detector_row": detector_row,
                "detector_row_label": row_label,
                "stabilizer": f"S{stabilizer}",
                "hit_rate": float(rates[detector_row, stabilizer]),
            }
            for stabilizer in range(stabilizers)
        )
    return rows


def _two_point_correlation(
    detector_flat: NDArray[np.int8],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    if detector_flat.size == 0:
        return np.empty((0, 0), dtype=float)
    detector_values = detector_flat.astype(float)
    means = detector_values.mean(axis=0)
    num_detectors = detector_values.shape[1]
    matrix = np.full((num_detectors, num_detectors), np.nan, dtype=float)
    for i in range(num_detectors):
        xi = means[i]
        for j in range(i + 1, num_detectors):
            xj = means[j]
            xij = float(np.mean(detector_values[:, i] * detector_values[:, j]))
            denom = 1.0 - 2.0 * xi - 2.0 * xj + 4.0 * xij
            if abs(denom) < eps:
                denom = eps if denom >= 0 else -eps
            radicand = 1.0 - 4.0 * (xij - xi * xj) / denom
            value = 0.5 - 0.5 * np.sqrt(max(0.0, radicand))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def plot_repetition_code_analysis(
    results: Any,
    *,
    show: bool = True,
    qec_camp_order: bool = True,
) -> list[Any]:
    """
    Plot the repetition-code logical summary and detector correlations.

    This mirrors the notebook workflow that first plots logical error rate
    versus round count and then plots one two-point detector-correlation matrix
    for each analysis payload.
    """
    payloads = _analysis_payloads(results)
    figures: list[Any] = []
    summary_figure = plot_repetition_code_logical_summary(payloads, show=show)
    if summary_figure is not None:
        figures.append(summary_figure)
    figures.extend(
        plot_repetition_code_correlation(
            payload,
            show=show,
            qec_camp_order=qec_camp_order,
        )
        for payload in payloads
    )
    return figures


def plot_repetition_code_logical_summary(
    results: Any,
    *,
    show: bool = True,
) -> Any | None:
    """Plot logical error rate versus round count for one or more results."""
    payloads = _analysis_payloads(results)
    rows: list[tuple[str, str, int, float]] = []
    for payload in payloads:
        logical_error_rate = _payload_float(payload, "logical_error_rate")
        if logical_error_rate is None:
            continue
        rows.append(
            (
                str(payload.get("basis", "")),
                str(payload.get("initial", "")),
                _payload_int(payload, "rounds", default=0),
                logical_error_rate,
            )
        )
    if not rows:
        print("No logical-error analysis payloads were provided.")
        return None

    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for basis, initial in sorted({(row[0], row[1]) for row in rows}):
        group = sorted(
            (rounds, rate)
            for group_basis, group_initial, rounds, rate in rows
            if (group_basis, group_initial) == (basis, initial)
        )
        ax.plot(
            [rounds for rounds, _ in group],
            [rate for _, rate in group],
            marker="o",
            label=f"{basis}:{initial}",
        )
    ax.set_xlabel("num_round")
    ax.set_ylabel("logical error rate")
    ax.set_title("repetition code logical error rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_repetition_code_correlation(
    result: Any,
    *,
    show: bool = True,
    qec_camp_order: bool = True,
) -> Any:
    """
    Plot one two-point detector-correlation matrix.

    When ``qec_camp_order`` is True and ``detector_array`` is available, the
    plot uses the qec_camp-style order ``S0:r0, ..., S0:final, S1:r0, ...``.
    """
    payload = _analysis_payload(result)
    matrix, labels, block = _correlation_matrix_for_plot(
        payload,
        qec_camp_order=qec_camp_order,
    )
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0

    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    im = ax.imshow(matrix, cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title(
        "repetition code p_ij: "
        f"{payload.get('basis', '')}:{payload.get('initial', '')}, "
        f"rounds={payload.get('rounds', '')}"
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    if block > 0:
        for boundary in range(block, len(labels), block):
            ax.axhline(boundary - 0.5, color="black", lw=0.6, alpha=0.5)
            ax.axvline(boundary - 0.5, color="black", lw=0.6, alpha=0.5)
    fig.colorbar(im, ax=ax, label="p_ij")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _analysis_payloads(results: Any) -> list[Mapping[str, object]]:
    if isinstance(results, (Result, Mapping)):
        return [_analysis_payload(results)]
    return [_analysis_payload(result) for result in results]


def _analysis_payload(result: Any) -> Mapping[str, object]:
    if isinstance(result, Result):
        return result.data
    if isinstance(result, Mapping):
        return result
    raise TypeError("Expected a Result, a payload mapping, or a sequence of them.")


def _correlation_matrix_for_plot(
    payload: Mapping[str, object],
    *,
    qec_camp_order: bool,
) -> tuple[NDArray[np.float64], list[str], int]:
    detector_array = payload.get("detector_array")
    if qec_camp_order and detector_array is not None:
        detector_tensor = np.asarray(detector_array, dtype=np.int8)
        if detector_tensor.ndim == 3:
            _shots, rows, stabilizers = detector_tensor.shape
            detector_flat = detector_tensor.transpose(0, 2, 1).reshape(
                detector_tensor.shape[0],
                -1,
            )
            return (
                _two_point_correlation(detector_flat),
                _detector_labels_qec_camp(stabilizers=stabilizers, rows=rows),
                rows,
            )

    matrix = np.asarray(payload["correlation_matrix"], dtype=float)
    fallback_labels = [f"D{index}" for index in range(matrix.shape[0])]
    labels_obj = payload.get("detector_labels")
    labels = (
        [str(label) for label in labels_obj]
        if isinstance(labels_obj, Sequence) and not isinstance(labels_obj, str)
        else fallback_labels
    )
    block = max(_payload_int(payload, "distance", default=1) - 1, 0)
    return matrix, labels, block


def _detector_labels_qec_camp(*, stabilizers: int, rows: int) -> list[str]:
    labels: list[str] = []
    for stabilizer in range(stabilizers):
        labels.extend(f"S{stabilizer}:r{row}" for row in range(rows - 1))
        labels.append(f"S{stabilizer}:final")
    return labels


def _payload_int(
    payload: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    return default


def _payload_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _matplotlib_pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def _decode_observables(
    detector_flat: NDArray[np.int8],
    *,
    distance: int,
    rounds: int,
) -> NDArray[np.int8]:
    decoder = _DecoderGraph(distance=distance, rounds=rounds)
    return np.asarray([decoder.decode(row) for row in detector_flat], dtype=np.int8)


class _DecoderGraph:
    def __init__(self, *, distance: int, rounds: int) -> None:
        self.distance = int(distance)
        self.rounds = int(rounds)
        self.num_rows = self.rounds + 1
        self.num_stabilizers = self.distance - 1
        self.num_detectors = self.num_rows * self.num_stabilizers
        self.left_boundary = self.num_detectors
        self.right_boundary = self.num_detectors + 1
        self.graph = self._build_graph()
        self.shortest_paths = {
            node: self._shortest_paths_from(node)
            for node in range(self.num_detectors + 2)
        }
        self._match_cache: dict[tuple[int, ...], tuple[int, int]] = {}

    def decode(self, detector_row: NDArray[np.int8]) -> int:
        defects = tuple(int(index) for index in np.flatnonzero(detector_row))
        _, observable = self._match(defects)
        return int(observable)

    def _detector_index(self, row: int, stabilizer: int) -> int:
        return row * self.num_stabilizers + stabilizer

    def _build_graph(self) -> list[list[tuple[int, int]]]:
        graph: list[list[tuple[int, int]]] = [[] for _ in range(self.num_detectors + 2)]

        def add_edge(left: int, right: int, observable: int = 0) -> None:
            graph[left].append((right, observable))
            graph[right].append((left, observable))

        for row in range(self.num_rows):
            for data_index in range(self.distance):
                if data_index == 0:
                    add_edge(self.left_boundary, self._detector_index(row, 0), 1)
                elif data_index == self.distance - 1:
                    add_edge(
                        self._detector_index(row, self.num_stabilizers - 1),
                        self.right_boundary,
                        1,
                    )
                else:
                    add_edge(
                        self._detector_index(row, data_index - 1),
                        self._detector_index(row, data_index),
                        1,
                    )

        for stabilizer in range(self.num_stabilizers):
            for row in range(self.num_rows - 1):
                add_edge(
                    self._detector_index(row, stabilizer),
                    self._detector_index(row + 1, stabilizer),
                    0,
                )
            for measurement_round in range(self.rounds):
                first_row = measurement_round
                if measurement_round <= self.rounds - 3:
                    second_row = measurement_round + 2
                else:
                    second_row = self.rounds
                if first_row != second_row:
                    add_edge(
                        self._detector_index(first_row, stabilizer),
                        self._detector_index(second_row, stabilizer),
                        0,
                    )
        return graph

    def _shortest_paths_from(self, start: int) -> dict[int, tuple[int, int]]:
        queue = [start]
        best: dict[int, tuple[int, int]] = {start: (0, 0)}
        for node in queue:
            distance, observable = best[node]
            for neighbor, edge_observable in self.graph[node]:
                candidate = (distance + 1, observable ^ edge_observable)
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    queue.append(neighbor)
        return best

    def _match(self, defects: tuple[int, ...]) -> tuple[int, int]:
        cached = self._match_cache.get(defects)
        if cached is not None:
            return cached
        if not defects:
            return (0, 0)
        first = defects[0]
        rest = defects[1:]
        best = self._pair_with_boundary(first, rest, self.left_boundary)
        best = min(best, self._pair_with_boundary(first, rest, self.right_boundary))
        for index, other in enumerate(rest):
            remaining = rest[:index] + rest[index + 1 :]
            pair_cost, pair_obs = self.shortest_paths[first][other]
            rem_cost, rem_obs = self._match(remaining)
            best = min(best, (pair_cost + rem_cost, pair_obs ^ rem_obs))
        self._match_cache[defects] = best
        return best

    def _pair_with_boundary(
        self,
        defect: int,
        remaining: tuple[int, ...],
        boundary: int,
    ) -> tuple[int, int]:
        pair_cost, pair_obs = self.shortest_paths[defect][boundary]
        rem_cost, rem_obs = self._match(remaining)
        return pair_cost + rem_cost, pair_obs ^ rem_obs


__all__ = [
    "plot_repetition_code_analysis",
    "plot_repetition_code_correlation",
    "plot_repetition_code_logical_summary",
    "repetition_code",
]
