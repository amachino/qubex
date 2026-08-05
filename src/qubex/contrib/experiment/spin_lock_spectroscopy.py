"""Spin-lock spectroscopy sequence helpers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.contrib.experiment.chevron_matched_transform import (
    estimate_qubit_frequency_from_chevron,
)
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import PulseSchedule, Rect

__all__ = ["spin_lock_sequence", "spin_lock_spectroscopy"]

DEFAULT_SPIN_LOCK_FREQUENCY_RANGE = np.geomspace(0.01, 0.15, 16)
DEFAULT_DURATION_RANGE = np.geomspace(100, 200e3, 31)
DEFAULT_DRIVE_PHASE = 0.0
FINAL_PROJECTION_PHASE_CORRECTION_SIGN = -1.0
CHEVRON_REFERENCE_RABI_FREQUENCY = 0.0125
CHEVRON_REFERENCE_DETUNING_HALF_WIDTH = 0.05
CHEVRON_MAX_DETUNING_HALF_WIDTH = 0.2
CHEVRON_REFERENCE_TIME_MAX = 250.0
CHEVRON_DETUNING_POINTS = 41
CHEVRON_TIME_POINTS = 26
CHEVRON_OMEGA_RABI_POINTS = 128
MIN_CHEVRON_TIME_POINTS = 3
MIN_DECAY_FIT_POINTS = 3
QUEL1_AWG_MAX_FREQUENCY = 0.25  # GHz
QUEL1_SAMPLING_PERIOD_NS = 2.0
QUEL1_WORD_LENGTH_SAMPLES = 4
QUEL1_BLOCK_LENGTH_SAMPLES = 64

ArrayMap = dict[str, np.ndarray]
ChevronMetadataMap = dict[str, list[dict[str, Any]]]
FigureMap = dict[str, go.Figure]


@dataclass(frozen=True)
class _TargetAnalysis:
    raw_data: np.ndarray
    normalized_signal: np.ndarray
    population: np.ndarray
    relaxation_times: np.ndarray
    relaxation_time_errors: np.ndarray
    r2: np.ndarray
    fit_results: list[dict[str, Any]]
    sort_indices: np.ndarray
    figures: FigureMap


def spin_lock_sequence(
    exp: Experiment,
    *,
    target: str,
    duration: float,
    drive_amplitude: float,
    drive_detuning: float | None = None,
    drive_phase: float | None = None,
    initial_phase: float | None = None,
    final_phase: float | None = None,
) -> PulseSchedule:
    """
    Build a simple spin-lock pulse schedule for one target qubit.

    Sequence structure:

    1. Initial half-pi pulse.
    2. Continuous spin-lock drive.
    3. Final half-pi pulse for projection.

    Parameters
    ----------
    exp
        Experiment instance used to obtain calibrated half-pi pulses.
    target
        Target qubit label.
    duration
        Spin-lock drive duration in ns.
    drive_amplitude
        Spin-lock drive amplitude.
    drive_detuning
        Optional spin-lock drive detuning in GHz.
    drive_phase
        Optional spin-lock drive phase in radians. Defaults to ``0``.
    initial_phase
        Optional phase for the initial half-pi pulse in radians. Defaults to
        ``drive_phase - pi/2``.
    final_phase
        Optional base phase for the final half-pi pulse in radians. Defaults to
        ``drive_phase + pi/2``. When ``drive_detuning`` is nonzero, the final
        projection phase is adjusted by ``-2 * pi * drive_detuning * duration``
        so the projection tracks the detuned spin-lock drive frame.

    Returns
    -------
        PulseSchedule
        Pulse schedule containing the spin-lock sequence.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")
    if not np.isfinite(drive_amplitude):
        raise ValueError("drive_amplitude must be finite.")
    if abs(drive_amplitude) > 1:
        raise ValueError("drive_amplitude must not exceed 1.")
    if drive_detuning is not None and not np.isfinite(drive_detuning):
        raise ValueError("drive_detuning must be finite.")

    with PulseSchedule([target]) as schedule:
        _add_spin_lock_pulse(
            schedule,
            exp,
            target=target,
            duration=duration,
            drive_amplitude=drive_amplitude,
            drive_detuning=drive_detuning,
            drive_phase=drive_phase,
            initial_phase=initial_phase,
            final_phase=final_phase,
        )
    return schedule


def _resolve_phases(
    *,
    drive_phase: float | None,
    initial_phase: float | None,
    final_phase: float | None,
) -> tuple[float, float, float]:
    resolved_drive_phase = DEFAULT_DRIVE_PHASE if drive_phase is None else drive_phase
    resolved_initial_phase = (
        resolved_drive_phase - np.pi / 2 if initial_phase is None else initial_phase
    )
    resolved_final_phase = (
        resolved_drive_phase + np.pi / 2 if final_phase is None else final_phase
    )
    if not np.all(
        np.isfinite(
            [resolved_drive_phase, resolved_initial_phase, resolved_final_phase]
        )
    ):
        raise ValueError("drive_phase, initial_phase, and final_phase must be finite.")
    return resolved_drive_phase, resolved_initial_phase, resolved_final_phase


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _resolve_frequency_range(
    spin_lock_frequency_range: ArrayLike | None,
) -> np.ndarray:
    if spin_lock_frequency_range is None:
        spin_lock_frequency_range = DEFAULT_SPIN_LOCK_FREQUENCY_RANGE
    frequencies = np.asarray(spin_lock_frequency_range, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("spin_lock_frequency_range must be a non-empty 1D array.")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("spin_lock_frequency_range must contain finite values.")
    if np.any(frequencies <= 0):
        raise ValueError("spin_lock_frequency_range must contain positive values.")
    return frequencies


def _resolve_duration_range(
    exp: Experiment,
    duration_range: ArrayLike | None,
) -> np.ndarray:
    if duration_range is None:
        duration_range = DEFAULT_DURATION_RANGE
    durations = np.asarray(duration_range, dtype=np.float64)
    if durations.ndim != 1 or durations.size == 0:
        raise ValueError("duration_range must be a non-empty 1D array.")
    if not np.all(np.isfinite(durations)):
        raise ValueError("duration_range must contain finite values.")
    if np.any(durations <= 0):
        raise ValueError("duration_range must contain positive values.")

    sampling_period = exp.ctx.measurement.sampling_period
    discretized = exp.ctx.util.discretize_time_range(
        durations,
        sampling_period=sampling_period,
    )
    if discretized is None:
        raise ValueError("duration_range could not be discretized.")

    discretized = np.asarray(discretized, dtype=np.float64)
    if np.any(discretized <= 0):
        raise ValueError("duration_range contains values below the sampling grid.")
    discretized = np.unique(discretized)
    if discretized.size < MIN_DECAY_FIT_POINTS:
        raise ValueError(
            "duration_range must contain at least "
            f"{MIN_DECAY_FIT_POINTS} unique points after discretization."
        )
    return discretized


def _resolve_drive_detuning(drive_detuning: float | None) -> float:
    if drive_detuning is None:
        return 0.0
    resolved_drive_detuning = float(drive_detuning)
    if not np.isfinite(resolved_drive_detuning):
        raise ValueError("drive_detuning must be finite.")
    return resolved_drive_detuning


def _resolve_spin_lock_amplitudes(
    exp: Experiment,
    *,
    targets: list[str],
    spin_lock_frequency_range: np.ndarray,
) -> ArrayMap:
    amplitudes = {
        target: np.asarray(
            [
                exp.pulse.calc_control_amplitude(target, float(frequency))
                for frequency in spin_lock_frequency_range
            ],
            dtype=np.float64,
        )
        for target in targets
    }
    for target, values in amplitudes.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Spin-lock amplitudes for `{target}` must be finite.")
        if np.any(np.abs(values) > 1):
            raise ValueError(
                f"Spin-lock amplitude for `{target}` must not exceed 1. "
                "Reduce spin_lock_frequency_range."
            )
    return amplitudes


def _resolve_positive_integer(value: int | None, *, name: str, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _resolve_nonnegative_finite_float(
    value: float | None,
    *,
    name: str,
    default: float,
) -> float:
    if value is None:
        value = default
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _resolve_bool(value: bool | None, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError("Boolean options must be bool or None.")
    return value


def _nan_array_map(
    *,
    targets: list[str],
    shape: tuple[int, ...],
) -> ArrayMap:
    return {
        target: np.full(
            shape,
            np.nan,
            dtype=np.float64,
        )
        for target in targets
    }


def _scaled_chevron_ranges(
    exp: Experiment,
    *,
    expected_rabi_frequency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = expected_rabi_frequency / CHEVRON_REFERENCE_RABI_FREQUENCY
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("expected_rabi_frequency must be positive and finite.")

    detuning_half_width = min(
        CHEVRON_REFERENCE_DETUNING_HALF_WIDTH * scale,
        CHEVRON_MAX_DETUNING_HALF_WIDTH,
    )
    detuning_range = np.linspace(
        -detuning_half_width,
        detuning_half_width,
        CHEVRON_DETUNING_POINTS,
    )

    time_max = CHEVRON_REFERENCE_TIME_MAX / scale
    time_range = exp.ctx.util.discretize_time_range(
        np.linspace(0, time_max, CHEVRON_TIME_POINTS),
        sampling_period=exp.ctx.measurement.sampling_period,
    )
    if time_range is None:
        raise ValueError("chevron time_range could not be discretized.")
    time_range = np.unique(np.asarray(time_range, dtype=np.float64))
    if time_range.size < MIN_CHEVRON_TIME_POINTS:
        raise ValueError(
            "chevron time_range must contain at least "
            f"{MIN_CHEVRON_TIME_POINTS} unique points after discretization."
        )

    omega_rabi_range = (
        expected_rabi_frequency
        * 2
        * (10 ** np.linspace(0, 1, CHEVRON_OMEGA_RABI_POINTS) - 1)
        / 9
    )
    return detuning_range, time_range, omega_rabi_range


def _target_base_frequencies(exp: Experiment, targets: list[str]) -> dict[str, float]:
    return {target: float(exp.ctx.targets[target].frequency) for target in targets}


def _chevron_drive_frequency_range(
    *,
    base_frequency: float,
    detuning_range: np.ndarray,
) -> np.ndarray:
    return float(base_frequency) + np.asarray(detuning_range, dtype=np.float64)


def _chevron_awg_frequency_range(
    exp: Experiment,
    *,
    target: str,
    drive_frequency_range: np.ndarray,
) -> np.ndarray:
    experiment_system = exp.ctx.experiment_system
    target_info = experiment_system.get_target(target)
    nco_frequency = float(experiment_system.get_nco_frequency(target))

    if target_info.sideband == "L":
        return nco_frequency - drive_frequency_range
    return drive_frequency_range - nco_frequency


def _looks_like_quel1_constraint_profile(constraint_profile: Any) -> bool:
    if constraint_profile is None:
        return False

    strict_quel1_flags = all(
        bool(getattr(constraint_profile, name, False))
        for name in (
            "require_workaround_capture",
            "enforce_word_alignment",
            "enforce_block_alignment",
            "enforce_capture_spacing",
        )
    )
    if not strict_quel1_flags:
        return False

    try:
        sampling_period_ns = float(
            getattr(constraint_profile, "sampling_period_ns", np.nan)
        )
        word_length_samples = int(
            getattr(constraint_profile, "word_length_samples", -1)
        )
        block_length_samples = int(
            getattr(constraint_profile, "block_length_samples", -1)
        )
    except (TypeError, ValueError):
        return False

    sampling_period_matches = bool(
        np.isclose(sampling_period_ns, QUEL1_SAMPLING_PERIOD_NS)
    )
    return (
        sampling_period_matches
        and word_length_samples == QUEL1_WORD_LENGTH_SAMPLES
        and block_length_samples == QUEL1_BLOCK_LENGTH_SAMPLES
    )


def _resolve_awg_max_frequency(exp: Experiment) -> float | None:
    constraint_profile = getattr(exp.ctx.measurement, "constraint_profile", None)
    if constraint_profile is None:
        return None

    awg_max_frequency_hz = getattr(constraint_profile, "awg_max_frequency_hz", None)
    if awg_max_frequency_hz is not None:
        awg_max_frequency = float(awg_max_frequency_hz) * 1e-9
        if not np.isfinite(awg_max_frequency) or awg_max_frequency <= 0:
            raise ValueError("awg_max_frequency_hz must be positive and finite.")
        return awg_max_frequency

    if _looks_like_quel1_constraint_profile(constraint_profile):
        return QUEL1_AWG_MAX_FREQUENCY

    return None


def _validate_awg_frequency_ranges(
    exp: Experiment,
    *,
    targets: list[str],
    drive_frequency_ranges: ArrayMap,
    description: str,
) -> None:
    awg_max_frequency = _resolve_awg_max_frequency(exp)
    if awg_max_frequency is None:
        return

    for target in targets:
        drive_frequency_range = drive_frequency_ranges[target]
        awg_frequency_range = _chevron_awg_frequency_range(
            exp,
            target=target,
            drive_frequency_range=drive_frequency_range,
        )
        max_abs_awg_frequency = float(np.max(np.abs(awg_frequency_range)))
        if max_abs_awg_frequency <= awg_max_frequency + 1e-12:
            continue

        raise ValueError(
            f"{description} for `{target}` requires AWG modulation up to "
            f"{max_abs_awg_frequency * 1e3:.3f} MHz, exceeding "
            f"{awg_max_frequency * 1e3:.3f} MHz with the current NCO setting. "
            "Reduce spin_lock_frequency_range or retune the backend settings. "
            f"Drive frequency range is {np.min(drive_frequency_range):.9g} GHz to "
            f"{np.max(drive_frequency_range):.9g} GHz."
        )


def _validate_chevron_awg_frequency_range(
    exp: Experiment,
    *,
    targets: list[str],
    base_frequencies: dict[str, float],
    detuning_range: np.ndarray,
) -> None:
    drive_frequency_ranges = {
        target: _chevron_drive_frequency_range(
            base_frequency=base_frequencies[target],
            detuning_range=detuning_range,
        )
        for target in targets
    }
    _validate_awg_frequency_ranges(
        exp,
        targets=targets,
        drive_frequency_ranges=drive_frequency_ranges,
        description="Chevron detuning_range",
    )


def _constant_frequency_map(
    *,
    targets: list[str],
    values: dict[str, float],
    shape: tuple[int, ...],
) -> ArrayMap:
    return {
        target: np.full(
            shape,
            values[target],
            dtype=np.float64,
        )
        for target in targets
    }


def _effective_frequency_map(
    *,
    rabi_frequency_map: ArrayMap,
    drive_detuning: float,
) -> ArrayMap:
    return {
        target: np.sqrt(rabi_frequency**2 + drive_detuning**2)
        for target, rabi_frequency in rabi_frequency_map.items()
    }


def _final_projection_phase(
    *,
    final_phase: float,
    drive_detuning: float | None,
    duration: float,
) -> float:
    if drive_detuning is None:
        return final_phase

    phase_correction = _final_projection_phase_correction(
        drive_detuning=float(drive_detuning),
        duration=float(duration),
    )
    if not np.isfinite(phase_correction):
        raise ValueError("final projection phase correction must be finite.")
    return final_phase + phase_correction


def _final_projection_phase_correction(
    *,
    drive_detuning: float,
    duration: float,
) -> float:
    return (
        FINAL_PROJECTION_PHASE_CORRECTION_SIGN * 2.0 * np.pi * drive_detuning * duration
    )


def _final_projection_phase_correction_grid(
    *,
    frequency_offsets: np.ndarray,
    durations: np.ndarray,
) -> np.ndarray:
    return (
        FINAL_PROJECTION_PHASE_CORRECTION_SIGN
        * 2.0
        * np.pi
        * np.asarray(frequency_offsets, dtype=np.float64)[:, None]
        * np.asarray(durations, dtype=np.float64)[None, :]
    )


def _final_projection_phase_maps(
    *,
    targets: list[str],
    frequency_offset_map: ArrayMap,
    durations: np.ndarray,
    final_phase: float,
) -> tuple[ArrayMap, ArrayMap]:
    phase_corrections: ArrayMap = {}
    final_phases: ArrayMap = {}

    for target in targets:
        correction = _final_projection_phase_correction_grid(
            frequency_offsets=frequency_offset_map[target],
            durations=durations,
        )
        if not np.all(np.isfinite(correction)):
            raise ValueError(
                f"Final projection phase corrections for `{target}` must be finite."
            )
        phase_corrections[target] = correction
        final_phases[target] = final_phase + correction

    return phase_corrections, final_phases


def _estimate_spin_lock_parameters_with_chevron(
    exp: Experiment,
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    amplitude_map: ArrayMap,
    chevron_n_shots: int,
    shot_interval: float,
    return_chevron_figures: bool,
) -> tuple[
    ArrayMap,
    ArrayMap,
    ChevronMetadataMap,
    FigureMap,
]:
    base_frequencies = _target_base_frequencies(exp, targets)
    measured_qubit_frequencies = _nan_array_map(
        targets=targets,
        shape=requested_rabi_frequency_range.shape,
    )
    measured_rabi_frequencies = _nan_array_map(
        targets=targets,
        shape=requested_rabi_frequency_range.shape,
    )
    chevron_results: ChevronMetadataMap = {target: [] for target in targets}
    chevron_figures: FigureMap = {}

    for frequency_index, expected_rabi_frequency in enumerate(
        requested_rabi_frequency_range
    ):
        detuning_range, time_range, omega_rabi_range = _scaled_chevron_ranges(
            exp,
            expected_rabi_frequency=float(expected_rabi_frequency),
        )
        _validate_chevron_awg_frequency_range(
            exp,
            targets=targets,
            base_frequencies=base_frequencies,
            detuning_range=detuning_range,
        )
        with exp.ctx.util.no_output():
            chevron_result = estimate_qubit_frequency_from_chevron(
                exp,
                targets,
                detuning_range=detuning_range,
                time_range=time_range,
                frequencies=base_frequencies,
                amplitudes={
                    target: float(amplitude_map[target][frequency_index])
                    for target in targets
                },
                omega_rabi_range=omega_rabi_range,
                n_shots=chevron_n_shots,
                shot_interval=shot_interval,
                plot=False,
                save_image=False,
            )

        results = cast(dict[str, dict[str, Any]], chevron_result.data["results"])
        for target in targets:
            result = results[target]
            measured_qubit_frequencies[target][frequency_index] = float(
                result["omega_q"]
            )
            measured_rabi_frequencies[target][frequency_index] = float(
                result["omega_rabi"]
            )
            if not np.isfinite(measured_qubit_frequencies[target][frequency_index]):
                raise ValueError(
                    f"Chevron qubit frequency for `{target}` is not finite."
                )
            if (
                not np.isfinite(measured_rabi_frequencies[target][frequency_index])
                or measured_rabi_frequencies[target][frequency_index] <= 0
            ):
                raise ValueError(
                    f"Chevron Rabi frequency for `{target}` must be positive and finite."
                )
            chevron_results[target].append(
                {
                    "requested_rabi_frequency": float(expected_rabi_frequency),
                    "drive_amplitude": float(amplitude_map[target][frequency_index]),
                    "qubit_frequency": float(result["omega_q"]),
                    "rabi_frequency": float(result["omega_rabi"]),
                    "peak_background_rms_ratio": float(
                        result["peak_background_rms_ratio"]
                    ),
                    "detuning_range": detuning_range,
                    "time_range": time_range,
                    "omega_rabi_range": omega_rabi_range,
                }
            )

        if return_chevron_figures and chevron_result.figures is not None:
            for name, figure in chevron_result.figures.items():
                figure_name = (
                    f"chevron_{frequency_index:03d}_"
                    f"{expected_rabi_frequency * 1e3:.6f}_MHz_{name}"
                )
                chevron_figures[figure_name] = figure

    return (
        measured_qubit_frequencies,
        measured_rabi_frequencies,
        chevron_results,
        chevron_figures,
    )


def _add_spin_lock_pulse(
    schedule: PulseSchedule,
    exp: Experiment,
    *,
    target: str,
    duration: float,
    drive_amplitude: float,
    drive_detuning: float | None,
    drive_phase: float | None,
    initial_phase: float | None,
    final_phase: float | None,
) -> None:
    drive_phase, initial_phase, final_phase = _resolve_phases(
        drive_phase=drive_phase,
        initial_phase=initial_phase,
        final_phase=final_phase,
    )

    half_pi = exp.pulse.get_hpi_pulse(target)
    spin_lock_drive = Rect(
        duration=duration,
        amplitude=drive_amplitude,
    ).shifted(drive_phase)

    if drive_detuning is not None:
        spin_lock_drive = spin_lock_drive.detuned(drive_detuning)

    final_phase = _final_projection_phase(
        final_phase=final_phase,
        drive_detuning=drive_detuning,
        duration=duration,
    )

    schedule.add(target, half_pi.shifted(initial_phase))
    schedule.add(target, spin_lock_drive)
    schedule.add(target, half_pi.shifted(final_phase))


def _make_spin_lock_heatmap_figure(
    *,
    target: str,
    spin_lock_rabi_frequency_range: np.ndarray,
    duration_range: np.ndarray,
    population: np.ndarray,
) -> go.Figure:
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=spin_lock_rabi_frequency_range * 1e3,
            y=duration_range,
            z=population,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Spin-lock spectroscopy : {target}",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Spin-lock Rabi frequency (MHz)",
        xaxis_type="log",
        yaxis_title="Duration (ns)",
        yaxis_type="log",
        width=700,
        height=500,
    )
    return fig


def _make_spin_lock_relaxation_figure(
    *,
    target: str,
    spin_lock_rabi_frequency_range: np.ndarray,
    relaxation_times: np.ndarray,
    relaxation_time_errors: np.ndarray,
) -> go.Figure:
    relaxation_times_us = relaxation_times * 1e-3
    relaxation_time_errors_us = relaxation_time_errors * 1e-3
    upper_candidates = relaxation_times_us + np.where(
        np.isfinite(relaxation_time_errors_us),
        np.maximum(relaxation_time_errors_us, 0.0),
        0.0,
    )
    finite_upper_candidates = upper_candidates[np.isfinite(upper_candidates)]
    yaxis_upper = (
        1.05 * float(np.max(finite_upper_candidates))
        if finite_upper_candidates.size > 0 and np.max(finite_upper_candidates) > 0
        else 1.0
    )

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=spin_lock_rabi_frequency_range * 1e3,
            y=relaxation_times_us,
            error_y=dict(
                type="data",
                array=relaxation_time_errors_us,
                visible=True,
            ),
            mode="markers+lines",
            name=target,
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Spin-lock relaxation : {target}",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Spin-lock Rabi frequency (MHz)",
        xaxis_type="log",
        yaxis_title="Relaxation time (μs)",
        yaxis_range=[0, yaxis_upper],
        width=700,
        height=450,
    )
    return fig


def _frequency_sort_indices(frequency_range: np.ndarray) -> np.ndarray:
    if np.any(np.diff(frequency_range) < 0):
        return np.argsort(frequency_range)
    return np.arange(frequency_range.size)


def _make_fit_summary(
    *,
    frequency: float,
    frequency_index: int,
    requested_rabi_frequency_range: np.ndarray,
    effective_frequency_range: np.ndarray,
    drive_frequencies: np.ndarray,
    qubit_frequencies: np.ndarray,
    status: FitStatus | str,
    message: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    status_value = status.value if isinstance(status, FitStatus) else status
    return {
        "frequency": float(frequency),
        "rabi_frequency": float(frequency),
        "effective_frequency": float(effective_frequency_range[frequency_index]),
        "requested_rabi_frequency": float(
            requested_rabi_frequency_range[frequency_index]
        ),
        "drive_frequency": float(drive_frequencies[frequency_index]),
        "qubit_frequency": float(qubit_frequencies[frequency_index]),
        "status": status_value,
        "message": message,
        "data": data,
    }


def _fit_data_without_figure(fit_result: Any) -> dict[str, Any]:
    return {key: value for key, value in fit_result.data.items() if key != "fig"}


def _analyze_spin_lock_target(
    *,
    target: str,
    sweep_data: Any,
    durations: np.ndarray,
    requested_rabi_frequency_range: np.ndarray,
    rabi_frequency_range: np.ndarray,
    effective_frequency_range: np.ndarray,
    drive_frequencies: np.ndarray,
    qubit_frequencies: np.ndarray,
) -> _TargetAnalysis:
    raw = np.asarray(sweep_data.data, dtype=np.complex128).reshape(
        requested_rabi_frequency_range.size,
        durations.size,
    )
    measured = np.asarray(sweep_data.normalized, dtype=np.float64).reshape(
        requested_rabi_frequency_range.size,
        durations.size,
    )
    fit_population = 0.5 * (1 + measured)

    relaxation_times = np.full(
        requested_rabi_frequency_range.shape,
        np.nan,
        dtype=np.float64,
    )
    relaxation_time_errors = np.full(
        requested_rabi_frequency_range.shape,
        np.nan,
        dtype=np.float64,
    )
    r2_values = np.full(
        requested_rabi_frequency_range.shape,
        np.nan,
        dtype=np.float64,
    )
    fit_results: list[dict[str, Any]] = []
    figures: FigureMap = {}

    for frequency_index, frequency in enumerate(rabi_frequency_range):
        try:
            fit_result = fitting.fit_exp_decay(
                target=f"{target}_{frequency * 1e3:.6g}_MHz",
                x=durations,
                y=fit_population[frequency_index],
                plot=False,
                title="Spin-lock relaxation",
                xlabel="Duration (μs)",
                ylabel="Normalized signal",
                xaxis_type="log",
                yaxis_type="linear",
            )
        except Exception as exc:
            fit_results.append(
                _make_fit_summary(
                    frequency=float(frequency),
                    frequency_index=frequency_index,
                    requested_rabi_frequency_range=requested_rabi_frequency_range,
                    effective_frequency_range=effective_frequency_range,
                    drive_frequencies=drive_frequencies,
                    qubit_frequencies=qubit_frequencies,
                    status=FitStatus.ERROR,
                    message=str(exc),
                    data={},
                )
            )
            continue

        if fit_result.status is FitStatus.SUCCESS:
            relaxation_times[frequency_index] = float(fit_result["tau"])
            relaxation_time_errors[frequency_index] = float(fit_result["tau_err"])
            r2_values[frequency_index] = float(fit_result["r2"])

        fit_results.append(
            _make_fit_summary(
                frequency=float(frequency),
                frequency_index=frequency_index,
                requested_rabi_frequency_range=requested_rabi_frequency_range,
                effective_frequency_range=effective_frequency_range,
                drive_frequencies=drive_frequencies,
                qubit_frequencies=qubit_frequencies,
                status=fit_result.status,
                message=fit_result.message or "",
                data=_fit_data_without_figure(fit_result),
            )
        )
        if fit_result.figure is not None:
            figures[f"{target}_fit_{frequency_index:03d}_{frequency * 1e3:.6f}_MHz"] = (
                fit_result.figure
            )

    sort_indices = _frequency_sort_indices(rabi_frequency_range)
    figures[f"{target}_heatmap"] = _make_spin_lock_heatmap_figure(
        target=target,
        spin_lock_rabi_frequency_range=rabi_frequency_range[sort_indices],
        duration_range=durations,
        population=fit_population.T[:, sort_indices],
    )
    figures[f"{target}_relaxation"] = _make_spin_lock_relaxation_figure(
        target=target,
        spin_lock_rabi_frequency_range=rabi_frequency_range[sort_indices],
        relaxation_times=relaxation_times[sort_indices],
        relaxation_time_errors=relaxation_time_errors[sort_indices],
    )

    return _TargetAnalysis(
        raw_data=raw,
        normalized_signal=measured.T,
        population=fit_population.T,
        relaxation_times=relaxation_times,
        relaxation_time_errors=relaxation_time_errors,
        r2=r2_values,
        fit_results=fit_results,
        sort_indices=sort_indices,
        figures=figures,
    )


def _measure_spin_lock_grid(
    exp: Experiment,
    *,
    targets: list[str],
    durations: np.ndarray,
    requested_rabi_frequency_range: np.ndarray,
    amplitude_map: ArrayMap,
    frequency_offset_map: ArrayMap,
    drive_phase: float | None,
    initial_phase: float | None,
    final_phase: float | None,
    n_shots: int,
    shot_interval: float,
    enable_tqdm: bool,
) -> Any:
    sweep_points = [
        (frequency_index, duration_index)
        for frequency_index in range(requested_rabi_frequency_range.size)
        for duration_index in range(durations.size)
    ]

    def sequence(point_index: int) -> PulseSchedule:
        frequency_index, duration_index = sweep_points[int(point_index)]
        with PulseSchedule(targets) as schedule:
            for target in targets:
                _add_spin_lock_pulse(
                    schedule,
                    exp,
                    target=target,
                    duration=float(durations[duration_index]),
                    drive_amplitude=float(amplitude_map[target][frequency_index]),
                    drive_detuning=float(frequency_offset_map[target][frequency_index]),
                    drive_phase=drive_phase,
                    initial_phase=initial_phase,
                    final_phase=final_phase,
                )
        return schedule

    with exp.ctx.util.no_output():
        return exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(sweep_points)),
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            enable_tqdm=enable_tqdm,
            title="Spin-lock spectroscopy",
            xlabel="Sweep point",
            ylabel="Measured value",
        )


def spin_lock_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    spin_lock_frequency_range: ArrayLike | None = None,
    duration_range: ArrayLike | None = None,
    estimate_with_chevron: bool | None = None,
    chevron_n_shots: int | None = None,
    return_chevron_figures: bool | None = None,
    drive_detuning: float | None = None,
    drive_phase: float | None = None,
    initial_phase: float | None = None,
    final_phase: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    enable_tqdm: bool | None = None,
) -> Result:
    """
    Run a simple spin-lock spectroscopy measurement.

    This experiment prepares each qubit along the spin-lock drive axis, applies
    a continuous drive for each requested duration, and projects the state back
    for readout. The requested spin-lock frequencies are interpreted as Rabi
    rates in GHz and are converted to hardware amplitudes with
    ``exp.pulse.calc_control_amplitude(target, rabi_rate)``. When
    ``drive_detuning`` is nonzero, the effective lock frequency is also stored
    as ``sqrt(rabi_rate**2 + drive_detuning**2)``.

    All ``(spin_lock_frequency, duration)`` points are flattened into one
    ``sweep_parameter()`` call.

    Parameters
    ----------
    exp
        Experiment instance used for pulse generation, measurement, and fitting.
    targets
        Target qubit labels. When omitted, all qubits in ``exp.ctx.qubit_labels``
        are measured.
    spin_lock_frequency_range
        Spin-lock frequencies in GHz. These values are treated as desired Rabi
        rates. If omitted, ``np.geomspace(0.01, 0.15, 16)`` is used, i.e.
        10 MHz to 150 MHz.
    duration_range
        Spin-lock drive durations in ns. Values must be positive because the
        heatmap and fit figures use a log time axis. The range is discretized to
        ``exp.ctx.measurement.sampling_period``. If omitted,
        ``np.geomspace(100, 200e3, 31)`` is used.
    estimate_with_chevron
        Whether to run ``estimate_qubit_frequency_from_chevron()`` before the
        spin-lock sweep for each requested spin-lock frequency. When ``True``,
        the spin-lock pulse frequency is shifted to the AC-Stark-shifted qubit
        frequency estimated by chevron, and the figure x-axes use the measured
        Rabi frequencies from chevron. The chevron detuning half-width is scaled
        from the expected Rabi rate and capped at 200 MHz.
    chevron_n_shots
        Number of shots for each pre-measurement chevron sweep point. Used only
        when ``estimate_with_chevron=True`` and defaults to
        ``max(1, n_shots // 4)``.
    return_chevron_figures
        Whether to include the intermediate chevron measurement and transform
        figures in ``Result.figures``. Defaults to ``False`` so chevron
        pre-measurement figures do not enlarge the returned figure payload.
    drive_detuning
        Optional spin-lock drive detuning in GHz.
    drive_phase
        Spin-lock drive phase in radians. When omitted, ``0`` is used.
    initial_phase
        Initial half-pi pulse phase in radians. When omitted,
        ``drive_phase - pi/2`` is used.
    final_phase
        Final half-pi pulse phase in radians. When omitted,
        ``drive_phase + pi/2`` is used. The actual final projection phase is
        additionally corrected by the accumulated detuned-drive phase for each
        spin-lock duration.
    n_shots
        Number of shots per sweep point. Defaults to ``DEFAULT_SHOTS``.
    shot_interval
        Measurement interval. Defaults to ``DEFAULT_INTERVAL``.
    plot
        Whether to display the heatmap and relaxation-time figures.
    save_image
        Whether to save the heatmap and relaxation-time figures.
    enable_tqdm
        Whether to show progress bars inside ``sweep_parameter()``. Defaults to
        ``False`` to match the measurement service default.

    Returns
    -------
    Result
        Generic result containing measured arrays, fit summaries, and figures.
        Important payload keys are:

        ``requested_spin_lock_rabi_frequency_range``
            Requested spin-lock Rabi-frequency axis in GHz.
        ``measured_spin_lock_rabi_frequency_range``
            Per-target Rabi-frequency axis in GHz used for plotting and fit
            metadata. This matches the requested axis unless
            ``estimate_with_chevron=True``.
        ``spin_lock_qubit_frequencies``
            Per-target qubit frequencies in GHz used as the spin-lock frequency
            reference before adding ``drive_detuning``.
        ``spin_lock_drive_frequencies``
            Per-target microwave drive frequencies in GHz including
            ``drive_detuning``. When ``estimate_with_chevron=True``, these are
            AC-Stark-shifted qubit frequencies estimated by chevron plus
            ``drive_detuning``.
        ``spin_lock_frequency_offsets``
            Per-target detunings in GHz applied to the spin-lock pulse relative
            to the calibrated qubit frequencies.
        ``final_projection_phase_corrections``
            Per-target phase corrections in radians applied to the final
            half-pi pulse, with shape
            ``(len(requested_spin_lock_rabi_frequency_range), len(duration_range))``.
        ``final_projection_phases``
            Per-target final half-pi pulse phases in radians after applying
            ``final_projection_phase_corrections``.
        ``effective_spin_lock_frequency_range``
            Per-target effective lock-frequency axis in GHz including
            ``drive_detuning``.
        ``duration_range``
            Discretized duration axis in ns.
        ``drive_amplitudes``
            Per-target drive amplitudes calculated from the requested spin-lock
            frequencies.
        ``raw_data``
            Per-target raw IQ data with shape
            ``(len(requested_spin_lock_rabi_frequency_range), len(duration_range))``.
        ``population``
            Per-target heatmap data with shape
            ``(len(duration_range), len(requested_spin_lock_rabi_frequency_range))``.
        ``normalized_signal``
            Per-target normalized signal before converting to population.
        ``relaxation_times``
            Per-target fitted spin-lock relaxation times in ns.
        ``relaxation_time_errors``
            Per-target one-sigma fit errors for the relaxation times in ns.
        ``r2``
            Per-target fit R² values.
        ``fit_results``
            Per-frequency fit metadata and fit parameters.
        ``chevron_results``
            Per-frequency chevron metadata. Empty unless
            ``estimate_with_chevron=True``.
        ``return_chevron_figures``
            Whether intermediate chevron figures were included in
            ``Result.figures``.
        ``chevron_n_shots``
            Number of shots used for each chevron sweep point, or ``None`` when
            ``estimate_with_chevron=False``.
        ``spin_lock_rabi_sort_indices``
            Per-target indices used to sort measured Rabi frequencies for
            plotting. Measured arrays in the payload keep the acquisition order.

        The ``figures`` mapping contains ``"{target}_heatmap"`` and
        ``"{target}_relaxation"`` for each target, plus per-frequency fit figures
        named ``"{target}_fit_{frequency_index}_{frequency_mhz}_MHz"``.

    Notes
    -----
    - ``spin_lock_frequency_range`` is not a detuning sweep; it is the target
      Rabi rate of the locking drive. With ``estimate_with_chevron=True``, the
      figure x-axis uses the measured Rabi-rate axis from chevron; otherwise it
      uses this requested Rabi-rate axis.
    - The initial and final half-pi pulses are left at the calibrated qubit
      frequency. When the spin-lock drive is detuned from that frequency, the
      final half-pi pulse phase is shifted by
      ``-2 * pi * drive_detuning * duration`` to follow the detuned drive frame.
    - The implementation raises an error if any calculated drive amplitude has
      absolute value larger than 1.
    - If the active measurement constraint profile defines
      ``awg_max_frequency_hz``, chevron pre-measurements and the final spin-lock
      drive frequencies are checked against that AWG modulation limit using the
      current NCO setting. For current QuEL-1-style strict timing profiles that
      do not expose such a field, a local 250 MHz contrib fallback is used.
    - The fitted relaxation uses ``0.5 * (1 + normalized_signal)`` and
      ``fitting.fit_exp_decay()``, matching the sign convention used by the T2
      echo analysis path.
    """
    target_list = _normalize_targets(exp, targets)
    if len(target_list) == 0:
        raise ValueError("targets must contain at least one target.")

    n_shots = _resolve_positive_integer(
        n_shots,
        name="n_shots",
        default=DEFAULT_SHOTS,
    )
    shot_interval = _resolve_nonnegative_finite_float(
        shot_interval,
        name="shot_interval",
        default=DEFAULT_INTERVAL,
    )
    plot = _resolve_bool(plot, default=True)
    save_image = _resolve_bool(save_image, default=False)
    enable_tqdm = _resolve_bool(enable_tqdm, default=False)
    estimate_with_chevron = _resolve_bool(estimate_with_chevron, default=False)
    return_chevron_figures = _resolve_bool(return_chevron_figures, default=False)

    exp.pulse.validate_rabi_params(target_list)

    requested_rabi_frequency_range = _resolve_frequency_range(spin_lock_frequency_range)
    drive_detuning_value = _resolve_drive_detuning(drive_detuning)
    drive_phase, initial_phase, final_phase = _resolve_phases(
        drive_phase=drive_phase,
        initial_phase=initial_phase,
        final_phase=final_phase,
    )
    durations = _resolve_duration_range(exp, duration_range)
    amplitude_map = _resolve_spin_lock_amplitudes(
        exp,
        targets=target_list,
        spin_lock_frequency_range=requested_rabi_frequency_range,
    )
    base_frequencies = _target_base_frequencies(exp, target_list)
    chevron_results: ChevronMetadataMap = {target: [] for target in target_list}
    chevron_figures: FigureMap = {}
    chevron_n_shots_value: int | None = None
    if estimate_with_chevron:
        chevron_n_shots_value = _resolve_positive_integer(
            chevron_n_shots,
            name="chevron_n_shots",
            default=max(1, n_shots // 4),
        )
        (
            qubit_frequency_map,
            measured_rabi_frequency_map,
            chevron_results,
            chevron_figures,
        ) = _estimate_spin_lock_parameters_with_chevron(
            exp,
            targets=target_list,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            amplitude_map=amplitude_map,
            chevron_n_shots=chevron_n_shots_value,
            shot_interval=shot_interval,
            return_chevron_figures=return_chevron_figures,
        )
    else:
        qubit_frequency_map = _constant_frequency_map(
            targets=target_list,
            values=base_frequencies,
            shape=requested_rabi_frequency_range.shape,
        )
        measured_rabi_frequency_map = {
            target: requested_rabi_frequency_range.copy() for target in target_list
        }
    drive_frequency_map = {
        target: qubit_frequency_map[target] + drive_detuning_value
        for target in target_list
    }
    _validate_awg_frequency_ranges(
        exp,
        targets=target_list,
        drive_frequency_ranges=drive_frequency_map,
        description="Spin-lock drive frequencies",
    )
    frequency_offset_map = {
        target: drive_frequency_map[target] - base_frequencies[target]
        for target in target_list
    }
    (
        final_projection_phase_corrections,
        final_projection_phases,
    ) = _final_projection_phase_maps(
        targets=target_list,
        frequency_offset_map=frequency_offset_map,
        durations=durations,
        final_phase=final_phase,
    )
    effective_measured_frequency_map = _effective_frequency_map(
        rabi_frequency_map=measured_rabi_frequency_map,
        drive_detuning=drive_detuning_value,
    )

    sweep_result = _measure_spin_lock_grid(
        exp,
        targets=target_list,
        durations=durations,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        amplitude_map=amplitude_map,
        frequency_offset_map=frequency_offset_map,
        drive_phase=drive_phase,
        initial_phase=initial_phase,
        final_phase=final_phase,
        n_shots=n_shots,
        shot_interval=shot_interval,
        enable_tqdm=enable_tqdm,
    )

    normalized_signal: ArrayMap = {}
    population: ArrayMap = {}
    raw_data: ArrayMap = {}
    relaxation_times: ArrayMap = {}
    relaxation_time_errors: ArrayMap = {}
    r2_values: ArrayMap = {}
    fit_results: dict[str, list[dict[str, Any]]] = {}
    rabi_sort_indices: ArrayMap = {}
    figures: FigureMap = dict(chevron_figures)

    for target in target_list:
        analysis = _analyze_spin_lock_target(
            target=target,
            sweep_data=sweep_result.data[target],
            durations=durations,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            rabi_frequency_range=measured_rabi_frequency_map[target],
            effective_frequency_range=effective_measured_frequency_map[target],
            drive_frequencies=drive_frequency_map[target],
            qubit_frequencies=qubit_frequency_map[target],
        )

        raw_data[target] = analysis.raw_data
        normalized_signal[target] = analysis.normalized_signal
        population[target] = analysis.population
        relaxation_times[target] = analysis.relaxation_times
        relaxation_time_errors[target] = analysis.relaxation_time_errors
        r2_values[target] = analysis.r2
        fit_results[target] = analysis.fit_results
        rabi_sort_indices[target] = analysis.sort_indices
        figures.update(analysis.figures)
        heatmap = figures[f"{target}_heatmap"]
        relaxation = figures[f"{target}_relaxation"]

        if plot:
            heatmap.show(config=viz.get_config(filename=f"spin_lock_heatmap_{target}"))
            relaxation.show(
                config=viz.get_config(filename=f"spin_lock_relaxation_{target}")
            )
        if save_image:
            viz.save_figure(heatmap, name=f"spin_lock_heatmap_{target}")
            viz.save_figure(relaxation, name=f"spin_lock_relaxation_{target}")

    primary_figure = figures.get(f"{target_list[0]}_relaxation")
    return Result(
        data={
            "targets": target_list,
            "requested_spin_lock_rabi_frequency_range": requested_rabi_frequency_range,
            "measured_spin_lock_rabi_frequency_range": measured_rabi_frequency_map,
            "effective_spin_lock_frequency_range": effective_measured_frequency_map,
            "spin_lock_qubit_frequencies": qubit_frequency_map,
            "spin_lock_drive_frequencies": drive_frequency_map,
            "spin_lock_frequency_offsets": frequency_offset_map,
            "final_projection_phase_corrections": final_projection_phase_corrections,
            "final_projection_phases": final_projection_phases,
            "duration_range": durations,
            "drive_amplitudes": amplitude_map,
            "estimate_with_chevron": estimate_with_chevron,
            "chevron_n_shots": chevron_n_shots_value,
            "return_chevron_figures": return_chevron_figures,
            "chevron_results": chevron_results,
            "spin_lock_rabi_sort_indices": rabi_sort_indices,
            "raw_data": raw_data,
            "normalized_signal": normalized_signal,
            "population": population,
            "relaxation_times": relaxation_times,
            "relaxation_time_errors": relaxation_time_errors,
            "r2": r2_values,
            "fit_results": fit_results,
        },
        figure=primary_figure,
        figures=figures,
    )
