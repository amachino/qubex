"""Spin-lock spectroscopy sequence helpers."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import PulseSchedule, Rect

__all__ = ["spin_lock_sequence", "spin_lock_spectroscopy"]

DEFAULT_SPIN_LOCK_FREQUENCY_RANGE = np.geomspace(0.001, 0.2, 21)
DEFAULT_DURATION_RANGE = np.geomspace(100, 200e3, 31)
DEFAULT_DRIVE_PHASE = 0.0


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
        Optional phase for the final half-pi pulse in radians. Defaults to
        ``drive_phase + pi/2``.

    Returns
    -------
        PulseSchedule
        Pulse schedule containing the spin-lock sequence.
    """
    if duration <= 0:
        raise ValueError("duration must be positive.")
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
    return discretized


def _effective_spin_lock_frequency_range(
    spin_lock_rabi_frequency_range: np.ndarray,
    *,
    drive_detuning: float | None,
) -> np.ndarray:
    if drive_detuning is None:
        drive_detuning = 0.0
    drive_detuning = float(drive_detuning)
    if not np.isfinite(drive_detuning):
        raise ValueError("drive_detuning must be finite.")
    return np.sqrt(spin_lock_rabi_frequency_range**2 + drive_detuning**2)


def _resolve_spin_lock_amplitudes(
    exp: Experiment,
    *,
    targets: list[str],
    spin_lock_frequency_range: np.ndarray,
) -> dict[str, np.ndarray]:
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
    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=spin_lock_rabi_frequency_range * 1e3,
            y=relaxation_times * 1e-3,
            error_y=dict(
                type="data",
                array=relaxation_time_errors * 1e-3,
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
        yaxis_title="Relaxation time (us)",
        yaxis_rangemode="tozero",
        width=700,
        height=450,
    )
    return fig


def spin_lock_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    spin_lock_frequency_range: ArrayLike | None = None,
    duration_range: ArrayLike | None = None,
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
    ``sweep_parameter()`` call, following the same pattern as the chevron
    measurement helpers.

    Parameters
    ----------
    exp
        Experiment instance used for pulse generation, measurement, and fitting.
    targets
        Target qubit labels. When omitted, all qubits in ``exp.ctx.qubit_labels``
        are measured.
    spin_lock_frequency_range
        Spin-lock frequencies in GHz. These values are treated as desired Rabi
        rates. If omitted, ``np.geomspace(0.001, 0.2, 21)`` is used, i.e.
        1 MHz to 200 MHz.
    duration_range
        Spin-lock drive durations in ns. Values must be positive because the
        heatmap and fit figures use a log time axis. The range is discretized to
        ``exp.ctx.measurement.sampling_period``. If omitted,
        ``np.geomspace(100, 200e3, 31)`` is used.
    drive_detuning
        Optional spin-lock drive detuning in GHz.
    drive_phase
        Spin-lock drive phase in radians. When omitted, ``0`` is used.
    initial_phase
        Initial half-pi pulse phase in radians. When omitted,
        ``drive_phase - pi/2`` is used.
    final_phase
        Final half-pi pulse phase in radians. When omitted,
        ``drive_phase + pi/2`` is used.
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

        ``spin_lock_rabi_frequency_range``
            Requested spin-lock Rabi-frequency axis in GHz.
        ``effective_spin_lock_frequency_range``
            Effective lock-frequency axis in GHz including ``drive_detuning``.
        ``spin_lock_frequency_range``
            Alias of ``spin_lock_rabi_frequency_range`` kept for convenience.
        ``duration_range``
            Discretized duration axis in ns.
        ``drive_amplitudes``
            Per-target drive amplitudes calculated from the requested spin-lock
            frequencies.
        ``raw_data``
            Per-target raw IQ data with shape
            ``(len(spin_lock_frequency_range), len(duration_range))``.
        ``population``
            Per-target heatmap data with shape
            ``(len(duration_range), len(spin_lock_frequency_range))``.
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

        The ``figures`` mapping contains ``"{target}_heatmap"`` and
        ``"{target}_relaxation"`` for each target, plus per-frequency fit figures
        named ``"{target}_fit_{frequency_mhz}_MHz"``.

    Notes
    -----
    - ``spin_lock_frequency_range`` is not a detuning sweep; it is the target
      Rabi rate of the locking drive. The figure x-axis uses this Rabi-rate
      axis, not the detuning-corrected effective lock frequency.
    - The implementation raises an error if any calculated drive amplitude has
      absolute value larger than 1.
    - The fitted relaxation uses ``0.5 * (1 + normalized_signal)`` and
      ``fitting.fit_exp_decay()``, matching the sign convention used by the T2
      echo analysis path.
    """
    target_list = _normalize_targets(exp, targets)
    if len(target_list) == 0:
        raise ValueError("targets must contain at least one target.")

    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if enable_tqdm is None:
        enable_tqdm = False

    exp.pulse.validate_rabi_params(target_list)

    rabi_frequency_range = _resolve_frequency_range(spin_lock_frequency_range)
    effective_frequency_range = _effective_spin_lock_frequency_range(
        rabi_frequency_range,
        drive_detuning=drive_detuning,
    )
    durations = _resolve_duration_range(exp, duration_range)
    amplitude_map = _resolve_spin_lock_amplitudes(
        exp,
        targets=target_list,
        spin_lock_frequency_range=rabi_frequency_range,
    )

    sweep_points = [
        (frequency_index, duration_index)
        for frequency_index in range(rabi_frequency_range.size)
        for duration_index in range(durations.size)
    ]

    def sequence(point_index: int) -> PulseSchedule:
        frequency_index, duration_index = sweep_points[int(point_index)]
        with PulseSchedule(target_list) as schedule:
            for target in target_list:
                _add_spin_lock_pulse(
                    schedule,
                    exp,
                    target=target,
                    duration=float(durations[duration_index]),
                    drive_amplitude=float(amplitude_map[target][frequency_index]),
                    drive_detuning=drive_detuning,
                    drive_phase=drive_phase,
                    initial_phase=initial_phase,
                    final_phase=final_phase,
                )
        return schedule

    with exp.ctx.util.no_output():
        sweep_result = exp.measurement_service.sweep_parameter(
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

    normalized_signal: dict[str, np.ndarray] = {}
    population: dict[str, np.ndarray] = {}
    raw_data: dict[str, np.ndarray] = {}
    relaxation_times: dict[str, np.ndarray] = {}
    relaxation_time_errors: dict[str, np.ndarray] = {}
    r2_values: dict[str, np.ndarray] = {}
    fit_results: dict[str, list[dict[str, Any]]] = {}
    figures: dict[str, go.Figure] = {}

    for target in target_list:
        sweep_data = sweep_result.data[target]
        raw = np.asarray(sweep_data.data, dtype=np.complex128).reshape(
            rabi_frequency_range.size,
            durations.size,
        )
        measured = np.asarray(sweep_data.normalized, dtype=np.float64).reshape(
            rabi_frequency_range.size,
            durations.size,
        )
        fit_population = 0.5 * (1 + measured)
        # Plot with duration on the vertical axis and frequency on the horizontal axis.
        normalized_signal[target] = measured.T
        population[target] = fit_population.T
        raw_data[target] = raw

        target_t1rho = np.full(rabi_frequency_range.shape, np.nan, dtype=np.float64)
        target_t1rho_err = np.full(
            rabi_frequency_range.shape,
            np.nan,
            dtype=np.float64,
        )
        target_r2 = np.full(rabi_frequency_range.shape, np.nan, dtype=np.float64)
        target_fit_results: list[dict[str, Any]] = []

        for frequency_index, frequency in enumerate(rabi_frequency_range):
            try:
                fit_result = fitting.fit_exp_decay(
                    target=f"{target}_{frequency * 1e3:.6g}_MHz",
                    x=durations,
                    y=fit_population[frequency_index],
                    plot=False,
                    title="Spin-lock relaxation",
                    xlabel="Duration (us)",
                    ylabel="Normalized signal",
                    xaxis_type="log",
                    yaxis_type="linear",
                )
            except Exception as exc:
                target_fit_results.append(
                    {
                        "frequency": float(frequency),
                        "rabi_frequency": float(frequency),
                        "effective_frequency": float(
                            effective_frequency_range[frequency_index]
                        ),
                        "status": FitStatus.ERROR.value,
                        "message": str(exc),
                        "data": {},
                    }
                )
                continue

            if fit_result.status is FitStatus.SUCCESS:
                target_t1rho[frequency_index] = float(fit_result["tau"])
                target_t1rho_err[frequency_index] = float(fit_result["tau_err"])
                target_r2[frequency_index] = float(fit_result["r2"])
            target_fit_results.append(
                {
                    "frequency": float(frequency),
                    "rabi_frequency": float(frequency),
                    "effective_frequency": float(
                        effective_frequency_range[frequency_index]
                    ),
                    "status": fit_result.status.value,
                    "message": fit_result.message,
                    "data": {
                        key: value
                        for key, value in fit_result.data.items()
                        if key != "fig"
                    },
                }
            )
            if fit_result.figure is not None:
                figures[f"{target}_fit_{frequency * 1e3:.6f}_MHz"] = fit_result.figure

        relaxation_times[target] = target_t1rho
        relaxation_time_errors[target] = target_t1rho_err
        r2_values[target] = target_r2
        fit_results[target] = target_fit_results

        heatmap = _make_spin_lock_heatmap_figure(
            target=target,
            spin_lock_rabi_frequency_range=rabi_frequency_range,
            duration_range=durations,
            population=population[target],
        )
        relaxation = _make_spin_lock_relaxation_figure(
            target=target,
            spin_lock_rabi_frequency_range=rabi_frequency_range,
            relaxation_times=target_t1rho,
            relaxation_time_errors=target_t1rho_err,
        )
        figures[f"{target}_heatmap"] = heatmap
        figures[f"{target}_relaxation"] = relaxation

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
            "spin_lock_frequency_range": rabi_frequency_range,
            "spin_lock_rabi_frequency_range": rabi_frequency_range,
            "effective_spin_lock_frequency_range": effective_frequency_range,
            "duration_range": durations,
            "drive_amplitudes": amplitude_map,
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
