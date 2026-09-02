"""Contributed warm-up (temperature-sweep) characterization campaign helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

from qubex.experiment.models.result import Result

from .thermal_excitation_characterization import measure_thermal_excitation

if TYPE_CHECKING:
    from qubex.experiment import Experiment

PLANCK_CONSTANT_JS = 6.62607015e-34
BOLTZMANN_CONSTANT_JK = 1.380649e-23

WARMUP_QUBIT_STEPS = ("ramsey", "t1", "t2_echo", "thermal", "single_shot")
WARMUP_RESONATOR_STEPS = ("reflection",)
WARMUP_STEPS = WARMUP_QUBIT_STEPS + WARMUP_RESONATOR_STEPS

_CORE_QUBIT_STEPS = ("ramsey", "t1", "t2_echo")

DEFAULT_WARMUP_MAX_DURATION = 12 * 3600.0
DEFAULT_WARMUP_THERMAL_SHOTS = 2**16
DEFAULT_WARMUP_MAX_FREQUENCY_SHIFT = 0.05
DEFAULT_WARMUP_MAX_RESONATOR_SHIFT = 0.2
DEFAULT_WARMUP_MAX_CONSECUTIVE_FAILURES = 3

_LOG_FILE_NAME = "warmup_log.jsonl"
_SUMMARY_FILE_NAME = "summary.json"
_SINGLE_SHOT_DIR_NAME = "single_shot"

_CORE_REQUIREMENTS = ("ge_frequency", "ge_rabi_params", "ge_pi_pulse")
_STEP_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ramsey": _CORE_REQUIREMENTS,
    "t1": _CORE_REQUIREMENTS,
    "t2_echo": _CORE_REQUIREMENTS,
    "thermal": ("ge_frequency", "ge_pi_pulse", "ef_pi_pulse"),
    "single_shot": ("ge_frequency", "ge_pi_pulse"),
    "reflection": ("read_frequency",),
}


def effective_temperature(p_ex: float, frequency: float) -> float:
    """
    Convert a thermal excitation probability into an effective temperature.

    Assumes a thermal two-level population ratio
    ``p_ex / (1 - p_ex) = exp(-h f / k_B T)``.

    Parameters
    ----------
    p_ex : float
        Excited-state population. Must lie in (0, 0.5) for a positive
        temperature; values outside that range return NaN.
    frequency : float
        Qubit ge frequency in GHz.

    Returns
    -------
    float
        Effective temperature in K, or NaN when the population or the
        frequency does not define a positive temperature.
    """
    if not 0.0 < p_ex < 0.5 or not np.isfinite(frequency) or frequency <= 0.0:
        return float("nan")
    energy = PLANCK_CONSTANT_JS * frequency * 1e9
    return float(energy / (BOLTZMANN_CONSTANT_JK * np.log((1.0 - p_ex) / p_ex)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_jsonable(value: object) -> object:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, (float, np.floating, np.integer)):
        return float(value)
    return str(value)


def _finite_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result


def _safe_call(action: Callable[[], Any]) -> tuple[Any, str | None]:
    """Run one action and report its value or the representation of its error."""
    try:
        value = action()
    except Exception as error:
        return None, repr(error)
    return value, None


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _normalize_steps(steps: Collection[str] | None) -> list[str]:
    if steps is None:
        return list(WARMUP_STEPS)
    normalized: list[str] = []
    for step in steps:
        canonical = step.strip().lower()
        if canonical not in WARMUP_STEPS:
            raise ValueError(
                f"Unknown warm-up step `{step}`. Expected one of {sorted(WARMUP_STEPS)}."
            )
        if canonical in normalized:
            raise ValueError(f"Duplicate warm-up step `{step}`.")
        normalized.append(canonical)
    if len(normalized) == 0:
        raise ValueError("At least one warm-up step must be specified.")
    return normalized


def preflight_check(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    verbose: bool = True,
) -> Result:
    """
    Check offline whether targets carry the calibrations the warm-up needs.

    No hardware is accessed. The check inspects stored frequencies, Rabi
    parameters, and pi pulses, then reports which warm-up steps each target
    is ready for. Run it before the warm-up day and fix missing
    calibrations; run the campaign with ``max_cycles=1`` afterwards to
    validate the chain on hardware.

    Parameters
    ----------
    exp : Experiment
        Experiment instance whose calibration note is inspected.
    targets : Collection[str] or str, optional
        Target qubits. Defaults to every qubit in the experiment.
    verbose : bool, optional
        Whether to print a readiness table.

    Returns
    -------
    Result
        Mapping-style result with per-target check flags, ready and missing
        steps, and an ``all_ready`` flag.
    """
    target_list = _normalize_targets(exp, targets)
    rabi_params, _ = _safe_call(lambda: dict(exp.pulse.rabi_params))
    rabi_params = rabi_params or {}

    report: dict[str, dict[str, Any]] = {}
    for target in target_list:
        ge_label, _ = _safe_call(lambda target=target: exp.ctx.resolve_ge_label(target))
        ef_label, _ = _safe_call(lambda target=target: exp.ctx.resolve_ef_label(target))
        read_label, _ = _safe_call(
            lambda target=target: exp.ctx.resolve_read_label(target)
        )

        def _frequency_ok(label: str | None) -> bool:
            if label is None:
                return False
            value, error = _safe_call(lambda: exp.ctx.targets[label].frequency)
            return error is None and np.isfinite(_finite_float(value))

        def _pulse_ok(label: str | None) -> bool:
            if label is None:
                return False
            _, error = _safe_call(lambda: exp.pulse.x180(label))
            return error is None

        checks = {
            "ge_frequency": _frequency_ok(ge_label),
            "read_frequency": _frequency_ok(read_label),
            "ge_rabi_params": ge_label is not None and ge_label in rabi_params,
            "ge_pi_pulse": _pulse_ok(ge_label),
            "ef_pi_pulse": _pulse_ok(ef_label),
        }
        ready_steps = [
            step
            for step, requirements in _STEP_REQUIREMENTS.items()
            if all(checks[requirement] for requirement in requirements)
        ]
        missing_steps = [step for step in _STEP_REQUIREMENTS if step not in ready_steps]
        report[target] = {
            "checks": checks,
            "ready_steps": ready_steps,
            "missing_steps": missing_steps,
        }

    all_ready = all(len(entry["missing_steps"]) == 0 for entry in report.values())

    if verbose:
        print("Warm-up preflight check")
        print("-----------------------")
        for target, entry in report.items():
            flags = " ".join(
                f"{name}={'ok' if value else 'NG'}"
                for name, value in entry["checks"].items()
            )
            missing = ", ".join(entry["missing_steps"]) or "none"
            print(f"{target}: {flags}")
            print(f"    missing steps: {missing}")
        print(f"all ready: {all_ready}")

    return Result(data={"targets": report, "all_ready": all_ready})


def check_mux_isolation(
    exp: Experiment,
    forbidden_muxes: Collection[int | str],
    *,
    verbose: bool = True,
) -> Result:
    """
    Check that the experiment shares no qubits or boxes with forbidden muxes.

    Use this before touching hardware when other muxes on the same system
    are in use by a concurrent experiment. Box-level operations such as
    AWG/capture-unit resets and clock re-synchronization act on whole
    boxes, so a box shared with a forbidden mux would disturb that
    experiment even if none of its qubits is addressed. No hardware is
    accessed by this check.

    Parameters
    ----------
    exp : Experiment
        Experiment instance whose selected qubits and boxes are inspected.
    forbidden_muxes : Collection[int or str]
        Mux indices or labels that must not be touched. A mux missing from
        the system configuration raises, so the check fails closed.
    verbose : bool, optional
        Whether to print the isolation report.

    Returns
    -------
    Result
        Mapping-style result with the selected and forbidden qubits and
        boxes, the shared qubits and boxes, and an ``isolated`` flag that
        is True only when nothing is shared.
    """
    system = exp.ctx.experiment_system
    selected_qubits = list(exp.ctx.qubit_labels)
    selected_boxes = sorted(exp.ctx.box_ids)

    forbidden_qubits: list[str] = []
    for mux in forbidden_muxes:
        mux_object, error = _safe_call(lambda mux=mux: system.get_mux(mux))
        if error is not None:
            raise ValueError(
                f"Forbidden mux `{mux}` was not found in the system configuration: {error}"
            )
        forbidden_qubits.extend(resonator.qubit for resonator in mux_object.resonators)

    boxes, error = _safe_call(lambda: system.get_boxes_for_qubits(forbidden_qubits))
    if error is not None:
        raise ValueError(f"Could not resolve boxes for forbidden muxes: {error}")
    forbidden_boxes = sorted({box.id for box in boxes})

    shared_qubits = sorted(set(selected_qubits) & set(forbidden_qubits))
    shared_boxes = sorted(set(selected_boxes) & set(forbidden_boxes))
    isolated = len(shared_qubits) == 0 and len(shared_boxes) == 0

    if verbose:
        print("Mux isolation check")
        print("-------------------")
        print(f"forbidden muxes  : {list(forbidden_muxes)}")
        print(f"forbidden qubits : {forbidden_qubits}")
        print(f"forbidden boxes  : {forbidden_boxes}")
        print(f"selected qubits  : {selected_qubits}")
        print(f"selected boxes   : {selected_boxes}")
        print(f"shared qubits    : {shared_qubits or 'none'}")
        print(f"shared boxes     : {shared_boxes or 'none'}")
        print(f"isolated         : {isolated}")

    return Result(
        data={
            "selected_qubits": selected_qubits,
            "selected_boxes": selected_boxes,
            "forbidden_qubits": forbidden_qubits,
            "forbidden_boxes": forbidden_boxes,
            "shared_qubits": shared_qubits,
            "shared_boxes": shared_boxes,
            "isolated": isolated,
        }
    )


class _CampaignLog:
    """Append-only JSONL campaign log with a live summary file."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.log_path = output_dir / _LOG_FILE_NAME
        self.summary_path = output_dir / _SUMMARY_FILE_NAME
        self.records: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {"updated_at": _utc_now(), "targets": {}}
        self._started_at = time.time()

    def record(
        self,
        step: str,
        *,
        cycle: int,
        target: str | None = None,
        status: str = "ok",
        values: dict[str, object] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Append one structured record to the JSONL log."""
        entry: dict[str, Any] = {
            "time": _utc_now(),
            "elapsed": round(time.time() - self._started_at, 3),
            "cycle": cycle,
            "step": step,
            "status": status,
        }
        if target is not None:
            entry["target"] = target
        if values is not None:
            entry["values"] = {
                key: _to_jsonable(value) for key, value in values.items()
            }
        if error is not None:
            entry["error"] = error
        self.records.append(entry)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if target is not None and status == "ok" and values is not None:
            target_summary = self._summary["targets"].setdefault(target, {})
            target_summary[step] = {
                "time": entry["time"],
                "cycle": cycle,
                **entry["values"],
            }
        return entry

    def flush_summary(self, extra: dict[str, Any]) -> None:
        """Rewrite the live summary file with the latest per-target values."""
        self._summary["updated_at"] = _utc_now()
        self._summary.update(extra)
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(self._summary, file, ensure_ascii=False, indent=2)


def warmup_campaign(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    output_dir: Path | str | None = None,
    max_duration: float = DEFAULT_WARMUP_MAX_DURATION,
    max_cycles: int | None = None,
    cycle_interval: float = 0.0,
    stop_file: Path | str | None = None,
    steps: Collection[str] | None = None,
    n_shots: int | None = None,
    ramsey_time_range: ArrayLike | None = None,
    ramsey_detuning: float | None = None,
    t1_time_range: ArrayLike | None = None,
    t2_time_range: ArrayLike | None = None,
    thermal_shots: int = DEFAULT_WARMUP_THERMAL_SHOTS,
    single_shot_states: Literal[2, 3] = 2,
    single_shot_count: int | None = None,
    reflection_frequency_width: float | None = None,
    reflection_df: float | None = None,
    electrical_delay: float | None = None,
    track_frequency: bool = True,
    max_frequency_shift: float = DEFAULT_WARMUP_MAX_FREQUENCY_SHIFT,
    max_resonator_shift: float = DEFAULT_WARMUP_MAX_RESONATOR_SHIFT,
    max_consecutive_failures: int = DEFAULT_WARMUP_MAX_CONSECUTIVE_FAILURES,
    refresh_reference_points: bool = True,
    refresh_rabi_every: int | None = 5,
    plot: bool = False,
) -> Result:
    """
    Run an interleaved characterization loop during a fridge warm-up.

    Each cycle measures, per target: Ramsey (T2* and frequency tracking),
    T1, T2 echo, thermal excitation (with effective temperature), a
    single-shot state distribution snapshot, and the resonator reflection
    coefficient (f_r, kappa_ex, kappa_in). All scalar outcomes are appended
    to ``warmup_log.jsonl`` in ``output_dir`` together with UTC timestamps,
    so fridge thermometry can be merged offline by time.

    Every step is fault-isolated: a failing measurement is logged and the
    campaign continues. When the core coherence steps (Ramsey, T1, T2 echo)
    fail ``max_consecutive_failures`` cycles in a row for a target, that
    target is treated as lost for qubit steps and only its resonator
    continues to be tracked. The campaign ends when the duration or cycle
    budget is spent, when ``stop_file`` appears, or when every signal is
    lost.

    Parameters
    ----------
    exp : Experiment
        Connected experiment instance. Restrict targets at construction
        time (e.g. ``Experiment(muxes=[8])``) or via ``targets``.
    targets : Collection[str] or str, optional
        Target qubits. Defaults to every qubit in the experiment.
    output_dir : Path or str, optional
        Directory for logs, the live summary, and single-shot data.
        Defaults to ``./warmup_data/<UTC timestamp>``.
    max_duration : float, optional
        Campaign duration budget in seconds.
    max_cycles : int, optional
        Maximum number of cycles. Unlimited when None.
    cycle_interval : float, optional
        Minimum time between cycle starts in seconds. Cycles run
        back-to-back when 0.
    stop_file : Path or str, optional
        The campaign stops gracefully once this file exists. Defaults to
        ``<output_dir>/STOP``.
    steps : Collection[str], optional
        Subset of ``WARMUP_STEPS`` to run. Defaults to every step.
    n_shots : int, optional
        Shot count for the coherence measurements.
    ramsey_time_range, t1_time_range, t2_time_range : array-like, optional
        Sweep ranges in ns forwarded to the underlying experiments.
    ramsey_detuning : float, optional
        Ramsey detuning in GHz forwarded to ``ramsey_experiment``.
    thermal_shots : int, optional
        Shots per sequence for the thermal excitation measurement.
    single_shot_states : {2, 3}, optional
        Number of prepared basis states in the single-shot snapshot.
    single_shot_count : int, optional
        Shot count for the single-shot snapshot.
    reflection_frequency_width, reflection_df, electrical_delay : optional
        Sweep width and step in GHz and electrical delay in ns forwarded to
        ``measure_reflection_coefficient``. Passing ``electrical_delay``
        avoids re-measuring the delay every cycle.
    track_frequency : bool, optional
        Whether to re-anchor drive frequencies to the fitted Ramsey bare
        frequency of the previous cycle.
    max_frequency_shift : float, optional
        Largest accepted qubit frequency shift from the calibrated value in
        GHz when tracking frequencies.
    max_resonator_shift : float, optional
        Largest accepted resonator frequency shift from the calibrated
        value in GHz when re-centering reflection scans.
    max_consecutive_failures : int, optional
        Consecutive fully-failed cycles after which a target's qubit steps
        (or its resonator tracking) are abandoned.
    refresh_reference_points : bool, optional
        Whether to refresh |g> reference points at each cycle start.
    refresh_rabi_every : int, optional
        Refresh Rabi parameters on the first cycle and then every this many
        cycles. Disabled when None.
    plot : bool, optional
        Whether to show figures of the underlying measurements.

    Returns
    -------
    Result
        Mapping-style result with the output paths, per-target alive flags,
        final tracked frequencies, the stop reason, and every log record.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be positive.")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be at least 1.")
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1.")
    if refresh_rabi_every is not None and refresh_rabi_every < 1:
        raise ValueError("refresh_rabi_every must be at least 1.")

    normalized_steps = _normalize_steps(steps)
    qubit_steps = [step for step in normalized_steps if step in WARMUP_QUBIT_STEPS]
    core_steps = [step for step in qubit_steps if step in _CORE_QUBIT_STEPS]
    resonator_steps = [
        step for step in normalized_steps if step in WARMUP_RESONATOR_STEPS
    ]
    target_list = _normalize_targets(exp, targets)
    if len(target_list) == 0:
        raise ValueError("At least one target is required.")

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("warmup_data") / stamp
    output_path = Path(output_dir)
    stop_path = Path(stop_file) if stop_file is not None else output_path / "STOP"
    single_shot_dir = output_path / _SINGLE_SHOT_DIR_NAME

    log = _CampaignLog(output_path)

    ge_labels = {target: exp.ctx.resolve_ge_label(target) for target in target_list}
    read_labels = {target: exp.ctx.resolve_read_label(target) for target in target_list}
    reference_freqs = {
        target: _finite_float(exp.ctx.targets[ge_labels[target]].frequency)
        for target in target_list
    }
    reference_read_freqs = {
        target: _finite_float(exp.ctx.targets[read_labels[target]].frequency)
        for target in target_list
    }

    tracked_freqs: dict[str, float] = {}
    resonator_centers: dict[str, float] = {}
    qubit_alive = dict.fromkeys(target_list, len(qubit_steps) > 0)
    resonator_alive = dict.fromkeys(target_list, len(resonator_steps) > 0)
    qubit_failures = dict.fromkeys(target_list, 0)
    resonator_failures = dict.fromkeys(target_list, 0)

    log.record(
        "campaign_start",
        cycle=0,
        values={
            "targets": ", ".join(target_list),
            "steps": ", ".join(normalized_steps),
            "max_duration": max_duration,
            "cycle_interval": cycle_interval,
            "thermal_shots": thermal_shots,
            "track_frequency": track_frequency,
            "stop_file": str(stop_path),
        },
    )
    for target in target_list:
        log.record(
            "reference",
            cycle=0,
            target=target,
            values={
                "f_ge": reference_freqs[target],
                "f_read": reference_read_freqs[target],
            },
        )

    def _record_scalars(
        step: str,
        cycle: int,
        target: str,
        values: dict[str, object],
        required: Collection[str],
    ) -> bool:
        ok = all(np.isfinite(_finite_float(values.get(key))) for key in required)
        log.record(
            step,
            cycle=cycle,
            target=target,
            status="ok" if ok else "failed",
            values=values,
        )
        return ok

    def _record_error(step: str, cycle: int, targets: list[str], error: str) -> None:
        for target in targets:
            log.record(step, cycle=cycle, target=target, status="failed", error=error)

    def _run_side_effect(step: str, cycle: int, action: Callable[[], Any]) -> None:
        _, error = _safe_call(action)
        log.record(
            step,
            cycle=cycle,
            status="ok" if error is None else "failed",
            error=error,
        )

    def _run_ramsey(alive: list[str], cycle: int, success: dict[str, bool]) -> None:
        result, error = _safe_call(
            lambda: exp.ramsey_experiment(
                targets=alive,
                time_range=ramsey_time_range,
                detuning=ramsey_detuning,
                n_shots=n_shots,
                plot=plot,
                save_image=False,
            )
        )
        if error is not None:
            _record_error("ramsey", cycle, alive, error)
            return
        for target in alive:
            data = result.data.get(target)
            t2_star = _finite_float(getattr(data, "t2", float("nan")))
            bare_freq = _finite_float(getattr(data, "bare_freq", float("nan")))
            ok = _record_scalars(
                "ramsey",
                cycle,
                target,
                {"t2_star": t2_star, "bare_freq": bare_freq},
                required=["t2_star", "bare_freq"],
            )
            if ok:
                success[target] = True
                shift = bare_freq - reference_freqs[target]
                if track_frequency and abs(shift) <= max_frequency_shift:
                    tracked_freqs[ge_labels[target]] = bare_freq

    def _run_coherence(
        step: str, alive: list[str], cycle: int, success: dict[str, bool]
    ) -> None:
        if step == "t1":
            attribute = "t1"
            result, error = _safe_call(
                lambda: exp.t1_experiment(
                    targets=alive,
                    time_range=t1_time_range,
                    n_shots=n_shots,
                    plot=plot,
                    save_image=False,
                )
            )
        else:
            attribute = "t2"
            result, error = _safe_call(
                lambda: exp.t2_experiment(
                    targets=alive,
                    time_range=t2_time_range,
                    n_shots=n_shots,
                    plot=plot,
                    save_image=False,
                )
            )
        if error is not None:
            _record_error(step, cycle, alive, error)
            return
        for target in alive:
            data = result.data.get(target)
            value = _finite_float(getattr(data, attribute, float("nan")))
            ok = _record_scalars(
                step, cycle, target, {attribute: value}, required=[attribute]
            )
            if ok:
                success[target] = True

    def _run_thermal(alive: list[str], cycle: int) -> None:
        for target in alive:
            result, error = _safe_call(
                lambda target=target: measure_thermal_excitation(
                    exp,
                    target,
                    n_shots=thermal_shots,
                    plot=plot,
                )
            )
            if error is not None:
                _record_error("thermal", cycle, [target], error)
                continue
            p_ex = _finite_float(result.data["p_ex"])
            frequency = tracked_freqs.get(ge_labels[target], reference_freqs[target])
            t_eff = effective_temperature(p_ex, frequency)
            _record_scalars(
                "thermal",
                cycle,
                target,
                {
                    "p_ex": p_ex,
                    "t_eff": t_eff if np.isfinite(t_eff) else None,
                    "frequency": frequency,
                },
                required=["p_ex"],
            )

    def _run_single_shot(alive: list[str], cycle: int) -> None:
        results, error = _safe_call(
            lambda: exp.measure_state_distribution(
                targets=alive,
                n_states=single_shot_states,
                n_shots=single_shot_count,
                plot=plot,
            )
        )
        if error is not None:
            _record_error("single_shot", cycle, alive, error)
            return
        single_shot_dir.mkdir(parents=True, exist_ok=True)
        file_path = single_shot_dir / f"cycle_{cycle:05d}.npz"
        arrays = {
            f"{target}_state{state}": np.asarray(result.data[target].kerneled)
            for state, result in enumerate(results)
            for target in alive
            if target in result.data
        }
        np.savez(file_path, **arrays)
        for target in alive:
            log.record(
                "single_shot",
                cycle=cycle,
                target=target,
                values={"file": str(file_path), "n_states": single_shot_states},
            )

    def _run_reflection(target: str, cycle: int) -> bool:
        result, error = _safe_call(
            lambda: exp.measure_reflection_coefficient(
                target,
                center_frequency=resonator_centers.get(target),
                frequency_width=reflection_frequency_width,
                df=reflection_df,
                electrical_delay=electrical_delay,
                plot=plot,
                save_image=False,
            )
        )
        if error is not None:
            _record_error("reflection", cycle, [target], error)
            return False
        f_r = _finite_float(result.data["f_r"])
        ok = _record_scalars(
            "reflection",
            cycle,
            target,
            {
                "f_r": f_r,
                "kappa_ex": _finite_float(result.data["kappa_ex"]),
                "kappa_in": _finite_float(result.data["kappa_in"]),
            },
            required=["f_r"],
        )
        shift = f_r - reference_read_freqs[target]
        if ok and np.isfinite(shift) and abs(shift) <= max_resonator_shift:
            resonator_centers[target] = f_r
        return ok

    stop_reason = "max_duration"
    started_at = time.monotonic()
    cycle = 0
    try:
        while True:
            if stop_path.exists():
                stop_reason = "stop_file"
                break
            if max_cycles is not None and cycle >= max_cycles:
                stop_reason = "max_cycles"
                break
            if time.monotonic() - started_at >= max_duration:
                stop_reason = "max_duration"
                break
            if not any(qubit_alive.values()) and not any(resonator_alive.values()):
                stop_reason = "all_signals_lost"
                break

            cycle += 1
            cycle_started_at = time.monotonic()
            alive = [target for target in target_list if qubit_alive[target]]
            log.record(
                "cycle_start",
                cycle=cycle,
                values={"alive_qubits": ", ".join(alive) or "none"},
            )

            core_success = dict.fromkeys(alive, False)
            if len(alive) > 0:
                if refresh_reference_points:
                    _run_side_effect(
                        "reference_points",
                        cycle,
                        lambda alive=alive: exp.obtain_reference_points(alive),
                    )
                if refresh_rabi_every is not None and (
                    (cycle - 1) % refresh_rabi_every == 0
                ):
                    _run_side_effect(
                        "rabi_refresh",
                        cycle,
                        lambda alive=alive: exp.obtain_rabi_params(alive, plot=plot),
                    )
                frequencies = dict(tracked_freqs) if track_frequency else None
                with exp.modified_frequencies(frequencies or None):
                    for step in qubit_steps:
                        if step == "ramsey":
                            _run_ramsey(alive, cycle, core_success)
                        elif step in ("t1", "t2_echo"):
                            _run_coherence(step, alive, cycle, core_success)
                        elif step == "thermal":
                            _run_thermal(alive, cycle)
                        elif step == "single_shot":
                            _run_single_shot(alive, cycle)

            for target in alive:
                if core_success[target] or len(core_steps) == 0:
                    qubit_failures[target] = 0
                else:
                    qubit_failures[target] += 1
                    if qubit_failures[target] >= max_consecutive_failures:
                        qubit_alive[target] = False
                        log.record("qubit_lost", cycle=cycle, target=target)

            for target in target_list:
                if not resonator_alive[target]:
                    continue
                if _run_reflection(target, cycle):
                    resonator_failures[target] = 0
                else:
                    resonator_failures[target] += 1
                    if resonator_failures[target] >= max_consecutive_failures:
                        resonator_alive[target] = False
                        log.record("resonator_lost", cycle=cycle, target=target)

            log.flush_summary(
                {
                    "cycle": cycle,
                    "qubit_alive": dict(qubit_alive),
                    "resonator_alive": dict(resonator_alive),
                    "tracked_frequencies": dict(tracked_freqs),
                    "resonator_centers": dict(resonator_centers),
                }
            )

            remaining = cycle_interval - (time.monotonic() - cycle_started_at)
            while remaining > 0 and not stop_path.exists():
                time.sleep(min(remaining, 5.0))
                remaining = cycle_interval - (time.monotonic() - cycle_started_at)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"

    log.record("campaign_end", cycle=cycle, values={"stop_reason": stop_reason})
    log.flush_summary({"stop_reason": stop_reason})

    return Result(
        data={
            "output_dir": str(output_path),
            "log_path": str(log.log_path),
            "summary_path": str(log.summary_path),
            "n_cycles": cycle,
            "stop_reason": stop_reason,
            "targets": target_list,
            "qubit_alive": dict(qubit_alive),
            "resonator_alive": dict(resonator_alive),
            "tracked_frequencies": dict(tracked_freqs),
            "resonator_centers": dict(resonator_centers),
            "records": list(log.records),
        },
    )


def load_warmup_log(path: Path | str) -> list[dict[str, Any]]:
    """
    Load warm-up campaign records from a JSONL log file.

    Parameters
    ----------
    path : Path or str
        Path to ``warmup_log.jsonl`` or to the campaign output directory.

    Returns
    -------
    list[dict]
        Parsed records in file order.
    """
    log_path = Path(path)
    if log_path.is_dir():
        log_path = log_path / _LOG_FILE_NAME
    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            content = line.strip()
            if content:
                records.append(json.loads(content))
    return records


_PLOT_SPECS: dict[str, tuple[str, str, float, str, str]] = {
    "t1": ("t1", "t1", 1e-3, "T1 (μs)", "linear"),
    "t2_echo": ("t2_echo", "t2", 1e-3, "T2 echo (μs)", "linear"),
    "t2_star": ("ramsey", "t2_star", 1e-3, "T2* (μs)", "linear"),
    "f_ge": ("ramsey", "bare_freq", 1.0, "Qubit frequency (GHz)", "linear"),
    "p_ex": ("thermal", "p_ex", 1.0, "Thermal excitation", "log"),
    "t_eff": ("thermal", "t_eff", 1e3, "Effective temperature (mK)", "linear"),
    "f_r": ("reflection", "f_r", 1.0, "Resonator frequency (GHz)", "linear"),
    "kappa_in": ("reflection", "kappa_in", 1e3, "κ_in (MHz)", "linear"),
    "kappa_ex": ("reflection", "kappa_ex", 1e3, "κ_ex (MHz)", "linear"),
}


def plot_warmup_log(
    records: Collection[dict[str, Any]] | Path | str,
    *,
    metrics: Collection[str] | None = None,
    targets: Collection[str] | None = None,
    save_dir: Path | str | None = None,
    plot: bool = True,
) -> dict[str, go.Figure]:
    """
    Plot warm-up campaign metrics against wall-clock time.

    Parameters
    ----------
    records : collection of dict, Path, or str
        Records from `load_warmup_log`, or a path to the log file or the
        campaign output directory.
    metrics : collection of str, optional
        Metric names to plot. Defaults to every metric with data. Valid
        names are ``t1``, ``t2_echo``, ``t2_star``, ``f_ge``, ``p_ex``,
        ``t_eff``, ``f_r``, ``kappa_in``, and ``kappa_ex``.
    targets : collection of str, optional
        Targets to include. Defaults to every target in the records.
    save_dir : Path or str, optional
        When given, each figure is also written there as an HTML file.
    plot : bool, optional
        Whether to show the figures.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Figures keyed by metric name.
    """
    if isinstance(records, (Path, str)):
        records = load_warmup_log(records)
    record_list = [record for record in records if record.get("status") == "ok"]

    if metrics is None:
        metric_list = list(_PLOT_SPECS)
    else:
        metric_list = list(metrics)
        for metric in metric_list:
            if metric not in _PLOT_SPECS:
                raise ValueError(
                    f"Unknown metric `{metric}`. Expected one of {sorted(_PLOT_SPECS)}."
                )

    figures: dict[str, go.Figure] = {}
    for metric in metric_list:
        step, key, scale, axis_title, yaxis_type = _PLOT_SPECS[metric]
        series: dict[str, tuple[list[str], list[float]]] = {}
        for record in record_list:
            if record.get("step") != step or "target" not in record:
                continue
            target = record["target"]
            if targets is not None and target not in targets:
                continue
            value = _finite_float(record.get("values", {}).get(key))
            if not np.isfinite(value):
                continue
            times, values = series.setdefault(target, ([], []))
            times.append(record["time"])
            values.append(value * scale)
        if len(series) == 0:
            continue
        fig = go.Figure()
        for target in sorted(series):
            times, values = series[target]
            fig.add_trace(
                go.Scatter(x=times, y=values, mode="lines+markers", name=target)
            )
        fig.update_layout(
            title=f"Warm-up : {axis_title}",
            xaxis_title="Time (UTC)",
            yaxis_title=axis_title,
            yaxis_type=yaxis_type,
            width=800,
            height=400,
        )
        figures[metric] = fig
        if plot:
            fig.show()
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path / f"warmup_{metric}.html")
    return figures
