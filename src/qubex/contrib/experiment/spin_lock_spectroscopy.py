"""Spin-lock spectroscopy sequence helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from tqdm import tqdm

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

DEFAULT_SPIN_LOCK_FREQUENCY_RANGE = (
    0.12 * np.tan(np.pi / 3 * np.linspace(0, 1, 15)) / np.sqrt(3)
)
DEFAULT_DURATION_RANGE = np.geomspace(100, 200e3, 31)
DEFAULT_DRIVE_PHASE = 0.0
CHEVRON_REFERENCE_RABI_FREQUENCY = 0.0125
CHEVRON_REFERENCE_DETUNING_HALF_WIDTH = 0.05
CHEVRON_NYQUIST_DETUNING_MARGIN = 0.8
CHEVRON_REFERENCE_TIME_MAX = 250.0
CHEVRON_DETUNING_POINTS = 41
CHEVRON_TIME_POINTS = 26
CHEVRON_OMEGA_RABI_POINTS = 128
MIN_DECAY_FIT_POINTS = 3
REQUESTED_RABI_FREQUENCIES_KEY = "requested_rabi_frequencies"
DRIVE_AMPLITUDES_KEY = "drive_amplitudes"
RABI_FREQUENCIES_KEY = "rabi_frequencies"
QUBIT_FREQUENCIES_KEY = "qubit_frequencies"
AC_STARK_SHIFTS_KEY = "ac_stark_shifts"
PEAK_BACKGROUND_RMS_RATIOS_KEY = "peak_background_rms_ratios"
CHEVRON_MEASURED_KEY = "chevron_measured"

ArrayMap = dict[str, np.ndarray]
SpinLockCalibrationEntry = dict[str, Any]
SpinLockCalibrationDataset = dict[str, SpinLockCalibrationEntry]
CalibrationSource = Literal["measured", "provided", "none"]
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
    figures: FigureMap


@dataclass(frozen=True)
class _SpinLockCalibration:
    drive_amplitude_map: ArrayMap
    qubit_frequency_map: ArrayMap
    rabi_frequency_map: ArrayMap
    calibration_dataset: SpinLockCalibrationDataset
    chevron_figures: FigureMap
    calibration_source: CalibrationSource
    chevron_n_shots: int | None


@dataclass(frozen=True)
class _SpinLockAnalysis:
    raw_data: ArrayMap
    normalized_signal: ArrayMap
    population: ArrayMap
    relaxation_times: ArrayMap
    relaxation_time_errors: ArrayMap
    r2: ArrayMap
    fit_results: dict[str, list[dict[str, Any]]]
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


def _validate_hpi_pulses(exp: Experiment, targets: list[str]) -> None:
    for target in targets:
        exp.pulse.get_hpi_pulse(target)


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
    if np.any(frequencies < 0):
        raise ValueError("spin_lock_frequency_range must contain nonnegative values.")
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

    sampling_period = _measurement_sampling_period_ns(exp)
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


def _measurement_sampling_period_ns(exp: Experiment) -> float:
    sampling_period = float(exp.ctx.measurement.sampling_period)
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("measurement sampling_period must be positive and finite.")
    return sampling_period


def _chevron_detuning_half_width_limit(exp: Experiment) -> float:
    return CHEVRON_NYQUIST_DETUNING_MARGIN * 0.5 / _measurement_sampling_period_ns(exp)


def _discretized_chevron_time_step(exp: Experiment, time_step: float) -> float:
    sampling_period = _measurement_sampling_period_ns(exp)
    discretized_time_step = exp.ctx.util.discretize_time_range(
        [time_step],
        sampling_period=sampling_period,
    )
    if discretized_time_step is None:
        raise ValueError("chevron time step could not be discretized.")

    discretized_time_step = np.asarray(discretized_time_step, dtype=np.float64)
    if discretized_time_step.size == 0:
        raise ValueError("chevron time step could not be discretized.")

    time_step = float(discretized_time_step[0])
    if not np.isfinite(time_step):
        raise ValueError("chevron time step must be finite after discretization.")
    return max(time_step, sampling_period)


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
        _chevron_detuning_half_width_limit(exp),
    )
    detuning_range = np.linspace(
        -detuning_half_width,
        detuning_half_width,
        CHEVRON_DETUNING_POINTS,
    )

    time_step = _discretized_chevron_time_step(
        exp,
        CHEVRON_REFERENCE_TIME_MAX / scale / (CHEVRON_TIME_POINTS - 1),
    )
    time_range = time_step * np.arange(CHEVRON_TIME_POINTS, dtype=np.float64)

    omega_rabi_range = (
        expected_rabi_frequency
        * 2
        * (10 ** np.linspace(0, 1, CHEVRON_OMEGA_RABI_POINTS) - 1)
        / 9
    )
    return detuning_range, time_range, omega_rabi_range


def _target_base_frequencies(exp: Experiment, targets: list[str]) -> dict[str, float]:
    return {target: float(exp.ctx.targets[target].frequency) for target in targets}


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
    return -2.0 * np.pi * drive_detuning * duration


def _final_projection_phase_correction_grid(
    *,
    frequency_offsets: np.ndarray,
    durations: np.ndarray,
) -> np.ndarray:
    return (
        -2.0
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


def _calibration_array(
    entry: Mapping[str, Any],
    *,
    key: str,
    target: str,
) -> np.ndarray:
    value = entry.get(key)
    if value is None:
        raise ValueError(f"spin_lock_calibration for `{target}` must contain `{key}`.")
    return np.asarray(value, dtype=np.float64)


def _optional_calibration_array(
    entry: Mapping[str, Any],
    *,
    key: str,
) -> np.ndarray | None:
    if key not in entry:
        return None
    return np.asarray(entry[key], dtype=np.float64)


def _optional_calibration_bool_array(
    entry: Mapping[str, Any],
    *,
    key: str,
    target: str,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if key not in entry:
        return None

    values = np.asarray(entry[key])
    if values.dtype != np.bool_:
        raise TypeError(f"spin_lock_calibration `{key}` for `{target}` must be bool.")

    _validate_calibration_shape(values, key=key, target=target, shape=shape)
    return values.astype(bool, copy=False)


def _validate_calibration_shape(
    values: np.ndarray,
    *,
    key: str,
    target: str,
    shape: tuple[int, ...],
) -> None:
    if values.shape != shape:
        raise ValueError(
            f"spin_lock_calibration for `{target}` has `{key}` shape "
            f"{values.shape}, but expected {shape}."
        )


def _validate_calibration_array(
    values: np.ndarray,
    *,
    key: str,
    target: str,
    shape: tuple[int, ...],
) -> None:
    _validate_calibration_shape(values, key=key, target=target, shape=shape)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"spin_lock_calibration `{key}` for `{target}` must be finite."
        )


def _frequency_axis_from_calibration(
    *,
    targets: list[str],
    spin_lock_calibration: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    first_target = targets[0]
    entry = spin_lock_calibration.get(first_target)
    if entry is None:
        raise ValueError(f"spin_lock_calibration must contain `{first_target}`.")
    if REQUESTED_RABI_FREQUENCIES_KEY in entry:
        return _calibration_array(
            entry,
            key=REQUESTED_RABI_FREQUENCIES_KEY,
            target=first_target,
        )
    return _calibration_array(entry, key=RABI_FREQUENCIES_KEY, target=first_target)


def _is_zero_rabi_frequency(frequency: float) -> bool:
    return bool(np.isclose(float(frequency), 0.0, rtol=0.0, atol=1e-15))


def _chevron_frequency_indices(requested_rabi_frequency_range: np.ndarray) -> list[int]:
    return [
        frequency_index
        for frequency_index, expected_rabi_frequency in enumerate(
            requested_rabi_frequency_range
        )
        if not _is_zero_rabi_frequency(float(expected_rabi_frequency))
    ]


def _is_valid_calibrated_rabi_frequency(
    *,
    requested_rabi_frequency: float,
    measured_rabi_frequency: float,
) -> bool:
    if not np.isfinite(measured_rabi_frequency):
        return False
    if measured_rabi_frequency > 0:
        return True
    return bool(
        _is_zero_rabi_frequency(requested_rabi_frequency)
        and np.isclose(
            measured_rabi_frequency,
            0.0,
            rtol=0.0,
            atol=1e-15,
        )
    )


def _make_spin_lock_calibration_dataset(
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    drive_amplitude_map: ArrayMap,
    qubit_frequency_map: ArrayMap,
    rabi_frequency_map: ArrayMap,
    base_frequencies: dict[str, float],
    peak_background_rms_ratio_map: ArrayMap | None = None,
    chevron_measured_map: dict[str, np.ndarray] | None = None,
) -> SpinLockCalibrationDataset:
    dataset: SpinLockCalibrationDataset = {}
    for target in targets:
        entry: SpinLockCalibrationEntry = {
            REQUESTED_RABI_FREQUENCIES_KEY: requested_rabi_frequency_range.copy(),
            DRIVE_AMPLITUDES_KEY: drive_amplitude_map[target].copy(),
            RABI_FREQUENCIES_KEY: rabi_frequency_map[target].copy(),
            QUBIT_FREQUENCIES_KEY: qubit_frequency_map[target].copy(),
            AC_STARK_SHIFTS_KEY: qubit_frequency_map[target] - base_frequencies[target],
        }
        if peak_background_rms_ratio_map is not None:
            entry[PEAK_BACKGROUND_RMS_RATIOS_KEY] = peak_background_rms_ratio_map[
                target
            ].copy()
        if chevron_measured_map is not None:
            entry[CHEVRON_MEASURED_KEY] = chevron_measured_map[target].copy()
        dataset[target] = entry
    return dataset


def _resolve_spin_lock_parameters_from_calibration(
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    spin_lock_calibration: Mapping[str, Mapping[str, Any]],
    base_frequencies: dict[str, float],
) -> tuple[ArrayMap, ArrayMap, ArrayMap, SpinLockCalibrationDataset]:
    shape = requested_rabi_frequency_range.shape
    drive_amplitude_map: ArrayMap = {}
    qubit_frequency_map: ArrayMap = {}
    rabi_frequency_map: ArrayMap = {}
    peak_background_rms_ratio_map: ArrayMap = {}
    chevron_measured_map: dict[str, np.ndarray] = {}

    for target in targets:
        entry = spin_lock_calibration.get(target)
        if entry is None:
            raise ValueError(f"spin_lock_calibration must contain `{target}`.")

        requested = _optional_calibration_array(
            entry,
            key=REQUESTED_RABI_FREQUENCIES_KEY,
        )
        if requested is not None:
            _validate_calibration_array(
                requested,
                key=REQUESTED_RABI_FREQUENCIES_KEY,
                target=target,
                shape=shape,
            )
            if not np.allclose(
                requested,
                requested_rabi_frequency_range,
                rtol=1e-6,
                atol=1e-12,
            ):
                raise ValueError(
                    f"spin_lock_calibration for `{target}` does not match "
                    "spin_lock_frequency_range."
                )

        drive_amplitudes = _calibration_array(
            entry,
            key=DRIVE_AMPLITUDES_KEY,
            target=target,
        )
        rabi_frequencies = _calibration_array(
            entry,
            key=RABI_FREQUENCIES_KEY,
            target=target,
        )
        _validate_calibration_array(
            drive_amplitudes,
            key=DRIVE_AMPLITUDES_KEY,
            target=target,
            shape=shape,
        )
        _validate_calibration_array(
            rabi_frequencies,
            key=RABI_FREQUENCIES_KEY,
            target=target,
            shape=shape,
        )

        qubit_frequencies = _optional_calibration_array(
            entry,
            key=QUBIT_FREQUENCIES_KEY,
        )
        if qubit_frequencies is None:
            ac_stark_shifts = _calibration_array(
                entry,
                key=AC_STARK_SHIFTS_KEY,
                target=target,
            )
            _validate_calibration_array(
                ac_stark_shifts,
                key=AC_STARK_SHIFTS_KEY,
                target=target,
                shape=shape,
            )
            qubit_frequencies = ac_stark_shifts + base_frequencies[target]
        else:
            _validate_calibration_array(
                qubit_frequencies,
                key=QUBIT_FREQUENCIES_KEY,
                target=target,
                shape=shape,
            )

        if np.any(np.abs(drive_amplitudes) > 1):
            raise ValueError(
                f"spin_lock_calibration drive amplitudes for `{target}` "
                "must not exceed 1."
            )
        peak_background_rms_ratios = _optional_calibration_array(
            entry,
            key=PEAK_BACKGROUND_RMS_RATIOS_KEY,
        )
        if peak_background_rms_ratios is not None:
            _validate_calibration_shape(
                peak_background_rms_ratios,
                key=PEAK_BACKGROUND_RMS_RATIOS_KEY,
                target=target,
                shape=shape,
            )
            peak_background_rms_ratio_map[target] = peak_background_rms_ratios

        chevron_measured = _optional_calibration_bool_array(
            entry,
            key=CHEVRON_MEASURED_KEY,
            target=target,
            shape=shape,
        )
        if chevron_measured is not None:
            chevron_measured_map[target] = chevron_measured

        for frequency_index, rabi_frequency in enumerate(rabi_frequencies):
            if not _is_valid_calibrated_rabi_frequency(
                requested_rabi_frequency=float(
                    requested_rabi_frequency_range[frequency_index]
                ),
                measured_rabi_frequency=float(rabi_frequency),
            ):
                raise ValueError(
                    f"spin_lock_calibration Rabi frequencies for `{target}` "
                    "must be positive and finite."
                )

        drive_amplitude_map[target] = drive_amplitudes
        qubit_frequency_map[target] = qubit_frequencies
        rabi_frequency_map[target] = rabi_frequencies

    optional_peak_background_rms_ratio_map = (
        peak_background_rms_ratio_map
        if len(peak_background_rms_ratio_map) == len(targets)
        else None
    )
    optional_chevron_measured_map = (
        chevron_measured_map if len(chevron_measured_map) == len(targets) else None
    )
    calibration_dataset = _make_spin_lock_calibration_dataset(
        targets=targets,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        drive_amplitude_map=drive_amplitude_map,
        qubit_frequency_map=qubit_frequency_map,
        rabi_frequency_map=rabi_frequency_map,
        base_frequencies=base_frequencies,
        peak_background_rms_ratio_map=optional_peak_background_rms_ratio_map,
        chevron_measured_map=optional_chevron_measured_map,
    )
    return (
        drive_amplitude_map,
        qubit_frequency_map,
        rabi_frequency_map,
        calibration_dataset,
    )


def _estimate_spin_lock_parameters_with_chevron(
    exp: Experiment,
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    amplitude_map: ArrayMap,
    base_frequencies: dict[str, float],
    chevron_n_shots: int,
    shot_interval: float,
    return_chevron_figures: bool,
) -> tuple[
    ArrayMap,
    ArrayMap,
    SpinLockCalibrationDataset,
    FigureMap,
]:
    measured_qubit_frequencies = _nan_array_map(
        targets=targets,
        shape=requested_rabi_frequency_range.shape,
    )
    measured_rabi_frequencies = _nan_array_map(
        targets=targets,
        shape=requested_rabi_frequency_range.shape,
    )
    peak_background_rms_ratios = _nan_array_map(
        targets=targets,
        shape=requested_rabi_frequency_range.shape,
    )
    chevron_measured = {
        target: np.full(requested_rabi_frequency_range.shape, False, dtype=bool)
        for target in targets
    }
    chevron_figures: FigureMap = {}

    chevron_frequency_indices = _chevron_frequency_indices(
        requested_rabi_frequency_range
    )
    for frequency_index, expected_rabi_frequency in enumerate(
        requested_rabi_frequency_range
    ):
        if not _is_zero_rabi_frequency(float(expected_rabi_frequency)):
            continue

        for target in targets:
            measured_qubit_frequencies[target][frequency_index] = float(
                base_frequencies[target]
            )
            measured_rabi_frequencies[target][frequency_index] = 0.0

    progress = tqdm(
        chevron_frequency_indices,
        desc="Chevron pre-measurements",
        unit="freq",
    )
    for frequency_index in progress:
        expected_rabi_frequency = requested_rabi_frequency_range[frequency_index]
        detuning_range, time_range, omega_rabi_range = _scaled_chevron_ranges(
            exp,
            expected_rabi_frequency=float(expected_rabi_frequency),
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
            peak_background_rms_ratios[target][frequency_index] = float(
                result["peak_background_rms_ratio"]
            )
            chevron_measured[target][frequency_index] = True

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
        _make_spin_lock_calibration_dataset(
            targets=targets,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            drive_amplitude_map=amplitude_map,
            qubit_frequency_map=measured_qubit_frequencies,
            rabi_frequency_map=measured_rabi_frequencies,
            base_frequencies=base_frequencies,
            peak_background_rms_ratio_map=peak_background_rms_ratios,
            chevron_measured_map=chevron_measured,
        ),
        chevron_figures,
    )


def _make_nominal_spin_lock_calibration(
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    amplitude_map: ArrayMap,
    base_frequencies: dict[str, float],
    include_chevron_measured: bool,
) -> _SpinLockCalibration:
    qubit_frequency_map = _constant_frequency_map(
        targets=targets,
        values=base_frequencies,
        shape=requested_rabi_frequency_range.shape,
    )
    rabi_frequency_map = {
        target: requested_rabi_frequency_range.copy() for target in targets
    }
    chevron_measured_map = (
        {
            target: np.full(requested_rabi_frequency_range.shape, False, dtype=bool)
            for target in targets
        }
        if include_chevron_measured
        else None
    )
    return _SpinLockCalibration(
        drive_amplitude_map=amplitude_map,
        qubit_frequency_map=qubit_frequency_map,
        rabi_frequency_map=rabi_frequency_map,
        calibration_dataset=_make_spin_lock_calibration_dataset(
            targets=targets,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            drive_amplitude_map=amplitude_map,
            qubit_frequency_map=qubit_frequency_map,
            rabi_frequency_map=rabi_frequency_map,
            base_frequencies=base_frequencies,
            chevron_measured_map=chevron_measured_map,
        ),
        chevron_figures={},
        calibration_source="none",
        chevron_n_shots=None,
    )


def _resolve_spin_lock_calibration(
    exp: Experiment,
    *,
    targets: list[str],
    requested_rabi_frequency_range: np.ndarray,
    base_frequencies: dict[str, float],
    spin_lock_calibration: Mapping[str, Mapping[str, Any]] | None,
    estimate_with_chevron: bool,
    chevron_n_shots: int | None,
    n_shots: int,
    shot_interval: float,
    return_chevron_figures: bool,
) -> _SpinLockCalibration:
    if spin_lock_calibration is not None:
        (
            drive_amplitude_map,
            qubit_frequency_map,
            rabi_frequency_map,
            calibration_dataset,
        ) = _resolve_spin_lock_parameters_from_calibration(
            targets=targets,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            spin_lock_calibration=spin_lock_calibration,
            base_frequencies=base_frequencies,
        )
        return _SpinLockCalibration(
            drive_amplitude_map=drive_amplitude_map,
            qubit_frequency_map=qubit_frequency_map,
            rabi_frequency_map=rabi_frequency_map,
            calibration_dataset=calibration_dataset,
            chevron_figures={},
            calibration_source="provided",
            chevron_n_shots=None,
        )

    amplitude_map = _resolve_spin_lock_amplitudes(
        exp,
        targets=targets,
        spin_lock_frequency_range=requested_rabi_frequency_range,
    )

    if estimate_with_chevron:
        chevron_frequency_indices = _chevron_frequency_indices(
            requested_rabi_frequency_range
        )
        if len(chevron_frequency_indices) == 0:
            return _make_nominal_spin_lock_calibration(
                targets=targets,
                requested_rabi_frequency_range=requested_rabi_frequency_range,
                amplitude_map=amplitude_map,
                base_frequencies=base_frequencies,
                include_chevron_measured=True,
            )

        chevron_n_shots_value = _resolve_positive_integer(
            chevron_n_shots,
            name="chevron_n_shots",
            default=max(1, n_shots // 4),
        )
        (
            qubit_frequency_map,
            rabi_frequency_map,
            measured_calibration_dataset,
            chevron_figures,
        ) = _estimate_spin_lock_parameters_with_chevron(
            exp,
            targets=targets,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            amplitude_map=amplitude_map,
            base_frequencies=base_frequencies,
            chevron_n_shots=chevron_n_shots_value,
            shot_interval=shot_interval,
            return_chevron_figures=return_chevron_figures,
        )
        return _SpinLockCalibration(
            drive_amplitude_map=amplitude_map,
            qubit_frequency_map=qubit_frequency_map,
            rabi_frequency_map=rabi_frequency_map,
            calibration_dataset=measured_calibration_dataset,
            chevron_figures=chevron_figures,
            calibration_source="measured",
            chevron_n_shots=chevron_n_shots_value,
        )

    return _make_nominal_spin_lock_calibration(
        targets=targets,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        amplitude_map=amplitude_map,
        base_frequencies=base_frequencies,
        include_chevron_measured=False,
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


def _frequency_axis_range_mhz(frequency_range: np.ndarray) -> list[float]:
    frequencies_mhz = np.asarray(frequency_range, dtype=np.float64) * 1e3
    finite_frequencies_mhz = frequencies_mhz[np.isfinite(frequencies_mhz)]
    if finite_frequencies_mhz.size == 0:
        return [0.0, 1.0]

    upper = float(np.max(finite_frequencies_mhz))
    return [0.0, 1.05 * upper if upper > 0 else 1.0]


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
        xaxis_type="linear",
        xaxis_range=_frequency_axis_range_mhz(spin_lock_rabi_frequency_range),
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
        xaxis_type="linear",
        xaxis_range=_frequency_axis_range_mhz(spin_lock_rabi_frequency_range),
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
    rabi_frequency: float,
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
        "rabi_frequency": float(rabi_frequency),
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
    data = getattr(fit_result, "data", {})
    return {key: value for key, value in dict(data).items() if key != "fig"}


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
                    rabi_frequency=float(frequency),
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
                rabi_frequency=float(frequency),
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
        figures=figures,
    )


def _analyze_spin_lock_targets(
    *,
    targets: list[str],
    sweep_result: Any,
    durations: np.ndarray,
    requested_rabi_frequency_range: np.ndarray,
    rabi_frequency_map: ArrayMap,
    effective_frequency_map: ArrayMap,
    drive_frequency_map: ArrayMap,
    qubit_frequency_map: ArrayMap,
    initial_figures: FigureMap,
) -> _SpinLockAnalysis:
    raw_data: ArrayMap = {}
    normalized_signal: ArrayMap = {}
    population: ArrayMap = {}
    relaxation_times: ArrayMap = {}
    relaxation_time_errors: ArrayMap = {}
    r2_values: ArrayMap = {}
    fit_results: dict[str, list[dict[str, Any]]] = {}
    figures: FigureMap = dict(initial_figures)

    for target in targets:
        analysis = _analyze_spin_lock_target(
            target=target,
            sweep_data=sweep_result.data[target],
            durations=durations,
            requested_rabi_frequency_range=requested_rabi_frequency_range,
            rabi_frequency_range=rabi_frequency_map[target],
            effective_frequency_range=effective_frequency_map[target],
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
        figures.update(analysis.figures)

    return _SpinLockAnalysis(
        raw_data=raw_data,
        normalized_signal=normalized_signal,
        population=population,
        relaxation_times=relaxation_times,
        relaxation_time_errors=relaxation_time_errors,
        r2=r2_values,
        fit_results=fit_results,
        figures=figures,
    )


def _show_or_save_spin_lock_figures(
    *,
    targets: list[str],
    figures: FigureMap,
    plot: bool,
    save_image: bool,
) -> None:
    for target in targets:
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

    print(f"Starting spin-lock spectroscopy measurement ({len(sweep_points)} points).")
    with exp.ctx.util.no_output():
        return exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(sweep_points)),
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            title="Spin-lock spectroscopy",
            xlabel="Sweep point",
            ylabel="Measured value",
        )


def _make_spin_lock_result_data(
    *,
    targets: list[str],
    calibration: _SpinLockCalibration,
    requested_rabi_frequency_range: np.ndarray,
    effective_frequency_map: ArrayMap,
    durations: np.ndarray,
    drive_detuning: float,
    drive_frequency_map: ArrayMap,
    frequency_offset_map: ArrayMap,
    final_projection_phase_corrections: ArrayMap,
    final_projection_phases: ArrayMap,
    analysis: _SpinLockAnalysis,
) -> dict[str, Any]:
    return {
        "targets": targets,
        "calibration_source": calibration.calibration_source,
        "spin_lock_calibration": calibration.calibration_dataset,
        "axes": {
            "requested_rabi_frequencies": requested_rabi_frequency_range,
            "measured_rabi_frequencies": calibration.rabi_frequency_map,
            "effective_spin_lock_frequencies": effective_frequency_map,
            "duration_range": durations,
        },
        "drive": {
            "drive_detuning": drive_detuning,
            "drive_frequencies": drive_frequency_map,
            "frequency_offsets": frequency_offset_map,
        },
        "phase": {
            "final_projection_phase_corrections": final_projection_phase_corrections,
            "final_projection_phases": final_projection_phases,
        },
        "measurement": {
            "raw_data": analysis.raw_data,
            "normalized_signal": analysis.normalized_signal,
            "population": analysis.population,
        },
        "fit": {
            "relaxation_times": analysis.relaxation_times,
            "relaxation_time_errors": analysis.relaxation_time_errors,
            "r2": analysis.r2,
            "fit_results": analysis.fit_results,
        },
        "metadata": {
            "chevron_n_shots": calibration.chevron_n_shots,
        },
    }


def spin_lock_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    spin_lock_frequency_range: ArrayLike | None = None,
    duration_range: ArrayLike | None = None,
    spin_lock_calibration: Mapping[str, Mapping[str, Any]] | None = None,
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
) -> Result:
    """
    Run a simple spin-lock spectroscopy measurement.

    This experiment prepares each qubit along the spin-lock drive axis, applies
    a continuous drive for each requested duration, and projects the state back
    for readout. The requested spin-lock frequencies are interpreted as Rabi
    rates in GHz. Without ``spin_lock_calibration``, they are converted to
    hardware amplitudes with
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
        rates. If omitted and ``spin_lock_calibration`` is not supplied,
        ``0.12 * tan(pi / 3 * linspace(0, 1, 15)) / sqrt(3)`` is used,
        i.e. 0 MHz to 120 MHz with denser points at low frequencies. If omitted
        and ``spin_lock_calibration`` is supplied, the sweep axis is read from
        the calibration dataset.
    duration_range
        Spin-lock drive durations in ns. Values must be positive because the
        heatmap and fit figures use a log time axis. The range is discretized to
        ``exp.ctx.measurement.sampling_period``. If omitted,
        ``np.geomspace(100, 200e3, 31)`` is used.
    spin_lock_calibration
        Previously returned ``Result.data["spin_lock_calibration"]`` or a
        manually prepared calibration dataset. When supplied, these values are
        reused and chevron pre-measurements are skipped. Each target entry must
        contain ``drive_amplitudes`` and ``rabi_frequencies``, plus either
        ``qubit_frequencies`` or ``ac_stark_shifts``. If
        ``spin_lock_frequency_range`` is omitted, ``requested_rabi_frequencies``
        is used as the sweep axis when present; otherwise ``rabi_frequencies``
        from the first target is used. All per-target arrays must have the same
        shape and order as the resolved spin-lock frequency sweep. Optional
        ``chevron_measured`` entries must be bool arrays. Rabi parameters are
        still required for readout normalization even when this calibration is
        supplied.
    estimate_with_chevron
        Whether to run ``estimate_qubit_frequency_from_chevron()`` before the
        spin-lock sweep for each requested spin-lock frequency. When ``True``,
        the spin-lock pulse frequency is shifted to the AC-Stark-shifted qubit
        frequency estimated by chevron, and the figure x-axes use the measured
        Rabi frequencies from chevron. The chevron detuning half-width is scaled
        from the expected Rabi rate and capped at 80% of the Nyquist frequency
        implied by ``exp.ctx.measurement.sampling_period``. Defaults to
        ``True``. The zero-frequency point, if present, is kept in the
        spin-lock sweep but skipped in chevron pre-measurements. Ignored when
        ``spin_lock_calibration`` is supplied. If all requested frequencies are
        zero, no chevron measurement is run and ``calibration_source`` remains
        ``"none"``.
    chevron_n_shots
        Number of shots for each pre-measurement chevron sweep point. Used only
        when chevrons are measured in this call and defaults to
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

    Returns
    -------
    Result
        ``spin_lock_calibration``
            Per-target calibration dataset containing ``drive_amplitudes``,
            ``rabi_frequencies``, ``qubit_frequencies``, and
            ``ac_stark_shifts`` in the same order as the sweep axis. This value
            can be supplied to a later ``spin_lock_spectroscopy()`` call.
        ``calibration_source``
            ``"measured"``, ``"provided"``, or ``"none"``.
        ``axes``
            Requested, measured, and effective Rabi-frequency axes in GHz, plus
            the discretized duration axis in ns.
        ``drive``
            Per-target spin-lock drive frequencies, offsets, and the common
            ``drive_detuning`` in GHz.
        ``phase``
            Final projection phases and duration-dependent corrections.
        ``measurement``
            Raw IQ data, normalized signal, and population heatmap data.
        ``fit``
            Relaxation times, fit errors, R² values, and per-frequency fit
            summaries.
        ``metadata``
            Additional measurement metadata such as ``chevron_n_shots``.

        The ``figures`` mapping contains ``"{target}_heatmap"`` and
        ``"{target}_relaxation"`` for each target, plus per-frequency fit figures
        named ``"{target}_fit_{frequency_index}_{frequency_mhz}_MHz"``.

    Notes
    -----
    - ``spin_lock_frequency_range`` is not a detuning sweep; it is the target
      Rabi rate of the locking drive. With spin-lock calibration, either
      measured by chevron in this call or supplied through
      ``spin_lock_calibration``, the figure x-axis uses the calibrated Rabi-rate
      axis; otherwise it uses this requested Rabi-rate axis.
    - A zero-frequency point is a free-precession reference without a spin-lock
      drive. It is kept in the final sweep and plots, but chevron calibration is
      skipped for that point.
    - The initial and final half-pi pulses are left at the calibrated qubit
      frequency. When the spin-lock drive is detuned from that frequency, the
      final half-pi pulse phase is shifted by
      ``-2 * pi * spin_lock_frequency_offset * duration`` to follow the
      spin-lock drive frame, including AC-Stark-shifted frequency offsets when
      spin-lock calibration is used.
    - The implementation raises an error if any drive amplitude has
      absolute value larger than 1.
    - Chevron detuning ranges are capped to stay within 80% of the Nyquist
      frequency implied by the measurement sampling period.
    - Spin-lock durations and chevron time steps are discretized on
      ``exp.ctx.measurement.sampling_period``. Backend-specific word or block
      alignment, if required, is left to the measurement stack.
    - The fitted relaxation uses ``0.5 * (1 + normalized_signal)`` and
      ``fitting.fit_exp_decay()``, matching the sign convention used by the T2
      echo analysis path.
    """
    target_list = _normalize_targets(exp, targets)
    if len(target_list) == 0:
        raise ValueError("targets must contain at least one target.")
    _validate_hpi_pulses(exp, target_list)
    exp.pulse.validate_rabi_params(target_list)

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
    estimate_with_chevron = _resolve_bool(estimate_with_chevron, default=True)
    return_chevron_figures = _resolve_bool(return_chevron_figures, default=False)

    if spin_lock_frequency_range is None and spin_lock_calibration is not None:
        spin_lock_frequency_range = _frequency_axis_from_calibration(
            targets=target_list,
            spin_lock_calibration=spin_lock_calibration,
        )
    requested_rabi_frequency_range = _resolve_frequency_range(spin_lock_frequency_range)
    drive_detuning_value = _resolve_drive_detuning(drive_detuning)
    drive_phase, initial_phase, final_phase = _resolve_phases(
        drive_phase=drive_phase,
        initial_phase=initial_phase,
        final_phase=final_phase,
    )
    durations = _resolve_duration_range(exp, duration_range)
    base_frequencies = _target_base_frequencies(exp, target_list)
    calibration = _resolve_spin_lock_calibration(
        exp,
        targets=target_list,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        base_frequencies=base_frequencies,
        spin_lock_calibration=spin_lock_calibration,
        estimate_with_chevron=estimate_with_chevron,
        chevron_n_shots=chevron_n_shots,
        n_shots=n_shots,
        shot_interval=shot_interval,
        return_chevron_figures=return_chevron_figures,
    )

    drive_frequency_map = {
        target: calibration.qubit_frequency_map[target] + drive_detuning_value
        for target in target_list
    }
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
        rabi_frequency_map=calibration.rabi_frequency_map,
        drive_detuning=drive_detuning_value,
    )

    sweep_result = _measure_spin_lock_grid(
        exp,
        targets=target_list,
        durations=durations,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        amplitude_map=calibration.drive_amplitude_map,
        frequency_offset_map=frequency_offset_map,
        drive_phase=drive_phase,
        initial_phase=initial_phase,
        final_phase=final_phase,
        n_shots=n_shots,
        shot_interval=shot_interval,
    )

    analysis = _analyze_spin_lock_targets(
        targets=target_list,
        sweep_result=sweep_result,
        durations=durations,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        rabi_frequency_map=calibration.rabi_frequency_map,
        effective_frequency_map=effective_measured_frequency_map,
        drive_frequency_map=drive_frequency_map,
        qubit_frequency_map=calibration.qubit_frequency_map,
        initial_figures=calibration.chevron_figures,
    )
    _show_or_save_spin_lock_figures(
        targets=target_list,
        figures=analysis.figures,
        plot=plot,
        save_image=save_image,
    )

    primary_figure = analysis.figures.get(f"{target_list[0]}_relaxation")
    result_data = _make_spin_lock_result_data(
        targets=target_list,
        calibration=calibration,
        requested_rabi_frequency_range=requested_rabi_frequency_range,
        effective_frequency_map=effective_measured_frequency_map,
        durations=durations,
        drive_detuning=drive_detuning_value,
        drive_frequency_map=drive_frequency_map,
        frequency_offset_map=frequency_offset_map,
        final_projection_phase_corrections=final_projection_phase_corrections,
        final_projection_phases=final_projection_phases,
        analysis=analysis,
    )
    return Result(
        data=result_data,
        figure=primary_figure,
        figures=analysis.figures,
    )
