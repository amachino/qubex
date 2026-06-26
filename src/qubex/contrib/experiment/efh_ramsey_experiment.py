"""EF/FH Ramsey experiment helpers."""

from __future__ import annotations

from collections.abc import Collection
from typing import TypedDict

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    HPI_DURATION,
    HPI_RAMPTIME,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, FlatTop, PulseSchedule


class _SpectrumResult(TypedDict):
    fft_frequencies: np.ndarray
    fft_values: np.ndarray
    dc_suppression_alpha: float
    fft_values_dc_suppressed: np.ndarray
    fft_frequencies_shifted: np.ndarray
    fft_values_shifted: np.ndarray
    dc_suppressed_waveform: np.ndarray


class _RamseyFitResult(TypedDict):
    status: str
    params: np.ndarray
    fit_time_range: np.ndarray
    fit_curve: np.ndarray
    amplitudes: np.ndarray
    frequencies: np.ndarray
    frequency_errors: np.ndarray
    phases: np.ndarray
    t2_star: float
    t2_star_error: float
    r2: float


class _TransitionFrequencyFit(TypedDict):
    frequencies: np.ndarray
    errors: np.ndarray
    mean: float
    mean_error: float
    components: dict[str, dict[str, float]]


def fh_ramsey_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    resonant_frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    omega_rabis: dict[str, float] | None = None,
    resonant_frequency: float | None = None,
    amplitude: float | None = None,
    omega_rabi: float | None = None,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    zero_fill_factor: int | None = None,
    dc_suppression_width: float | None = None,
    peak_distance: float | None = None,
    n_peaks: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Measure FH Ramsey data and project the IQ trace onto its PCA axis.

    The pulse sequence is
    ``X_ge -> X_ef -> FH X90 -> wait T -> FH X90 -> X_ef -> X_ge -> readout``.
    """
    target_labels = _normalize_targets(exp, targets)
    if detuning is None:
        detuning = 0.002
    if duration is None:
        duration = HPI_DURATION
    if ramptime is None:
        ramptime = HPI_RAMPTIME
    if zero_fill_factor is None:
        zero_fill_factor = 8
    if dc_suppression_width is None:
        dc_suppression_width = 0.0005
    if peak_distance is None:
        peak_distance = 0.0001
    if n_peaks is None:
        n_peaks = 2
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if time_range is None:
        time_range = np.arange(0, 10001, 100)

    sampling_period = exp.ctx.measurement.sampling_period
    time_values = exp.ctx.util.discretize_time_range(
        np.asarray(time_range, dtype=np.float64),
        sampling_period=sampling_period,
    )
    if time_values is None:
        raise ValueError("time_range could not be discretized.")

    normalized_frequencies = _normalize_target_values(
        exp,
        target_labels,
        values=resonant_frequencies,
        scalar_value=resonant_frequency,
        value_name="resonant_frequency",
    )
    normalized_amplitudes = _normalize_target_values(
        exp,
        target_labels,
        values=amplitudes,
        scalar_value=amplitude,
        value_name="amplitude",
    )
    normalized_omega_rabis = _normalize_target_values(
        exp,
        target_labels,
        values=omega_rabis,
        scalar_value=omega_rabi,
        value_name="omega_rabi",
    )

    fh_x90_pulses = {
        target: _build_transition_x90_pulse(
            amplitude=normalized_amplitudes[target],
            omega_rabi=normalized_omega_rabis[target],
            duration=duration,
            ramptime=ramptime,
            sampling_period=sampling_period,
        )
        for target in target_labels
    }
    drive_frequencies = {
        target: normalized_frequencies[target] + detuning for target in target_labels
    }

    data = {}
    figures = {}
    for target in target_labels:
        qubit = exp.ctx.resolve_qubit_label(target)
        ge_label = exp.ctx.resolve_ge_label(target)
        ef_label = exp.ctx.resolve_ef_label(target)
        fh_label = exp.ctx.resolve_fh_label(target)
        fh_x90 = fh_x90_pulses[target]

        def ramsey_sequence(
            wait_time: int,
            ge_label: str = ge_label,
            ef_label: str = ef_label,
            fh_label: str = fh_label,
            fh_x90: FlatTop = fh_x90,
        ) -> PulseSchedule:
            with PulseSchedule() as ps:
                ps.add(ge_label, exp.pulse.x180(ge_label))
                ps.barrier()
                ps.add(ef_label, exp.pulse.x180(ef_label))
                ps.barrier()
                ps.add(fh_label, fh_x90)
                ps.add(fh_label, Blank(wait_time))
                ps.add(fh_label, fh_x90)
                ps.barrier()
                ps.add(ef_label, exp.pulse.x180(ef_label))
                ps.barrier()
                ps.add(ge_label, exp.pulse.x180(ge_label))
            return ps

        with exp.ctx.util.no_output():
            sweep_result = exp.measurement_service.sweep_parameter(
                sequence=ramsey_sequence,
                sweep_range=time_values,
                frequencies={target: drive_frequencies[target]},
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=plot,
            )

        iq = np.asarray(sweep_result.data[qubit].data, dtype=np.complex128)
        projection = _project_iq_pca(iq)
        spectrum = _calculate_spectrum(
            time_range=time_values,
            projected=projection["projected"],
            zero_fill_factor=zero_fill_factor,
            dc_suppression_width=dc_suppression_width,
        )
        peaks = _find_ramsey_spectrum_peaks(
            fft_frequencies=spectrum["fft_frequencies"],
            fft_values=spectrum["fft_values_dc_suppressed"],
            detuning=detuning,
            dc_suppression_width=dc_suppression_width,
            peak_distance=peak_distance,
            n_peaks=n_peaks,
        )
        fit = _fit_projected_ramsey(
            time_range=time_values,
            projected=spectrum["dc_suppressed_waveform"],
            peak_frequencies=peaks["frequencies"],
            n_peaks=n_peaks,
            detuning=detuning,
        )
        fh_frequency_fit = _calculate_transition_frequency_fit(
            drive_frequency=drive_frequencies[target],
            fit_frequencies=fit["frequencies"],
            fit_frequency_errors=fit["frequency_errors"],
        )
        fig = _make_raw_ramsey_figure(
            target=target,
            time_range=time_values,
            projected=projection["projected"],
            resonant_frequency=normalized_frequencies[target],
            drive_frequency=drive_frequencies[target],
            amplitude=fh_x90_pulses[target].amplitude,
        )
        filtered_fig = _make_filtered_ramsey_figure(
            target=target,
            time_range=time_values,
            filtered_waveform=spectrum["dc_suppressed_waveform"],
            fit_time_range=fit["fit_time_range"],
            fit_curve=fit["fit_curve"],
        )
        spectrum_fig = _make_spectrum_figure(
            target=target,
            fft_frequencies=spectrum["fft_frequencies_shifted"],
            fft_values=spectrum["fft_values_shifted"],
            peak_frequencies=peaks["frequencies"],
            peak_values=spectrum["fft_values"][peaks["indices"]],
        )

        if plot:
            _print_ramsey_fit_result(
                target=target,
                fit=fit,
                transition_frequency_fit=fh_frequency_fit,
            )
            fig.show()
            spectrum_fig.show()
            filtered_fig.show()
        if save_image:
            viz.save_figure(
                fig,
                name=f"fh_ramsey_{target}",
                width=600,
                height=400,
            )
            viz.save_figure(
                filtered_fig,
                name=f"fh_ramsey_filtered_{target}",
                width=600,
                height=400,
            )
            viz.save_figure(
                spectrum_fig,
                name=f"fh_ramsey_spectrum_{target}",
                width=600,
                height=400,
            )

        data[target] = {
            "target": target,
            "qubit": qubit,
            "inputs": {
                "resonant_frequency": normalized_frequencies[target],
                "drive_frequency": drive_frequencies[target],
                "detuning": detuning,
                "chevron_amplitude": normalized_amplitudes[target],
                "omega_rabi": normalized_omega_rabis[target],
                "n_shots": n_shots,
                "shot_interval": shot_interval,
            },
            "traces": {
                "time_range": time_values,
                "iq": iq,
                "projected": projection["projected"],
                "dc_suppressed_waveform": spectrum["dc_suppressed_waveform"],
            },
            "pca": {
                "axis": projection["axis"],
                "center": projection["center"],
            },
            "spectrum": {
                "dc_suppression_width": dc_suppression_width,
                "dc_suppression_alpha": spectrum["dc_suppression_alpha"],
                "peak_distance": peak_distance,
                "peak_frequencies": peaks["frequencies"],
                "peak_amplitudes": peaks["amplitudes"],
            },
            "fit": {
                "status": fit["status"],
                "params": fit["params"],
                "time_range": fit["fit_time_range"],
                "curve": fit["fit_curve"],
                "amplitudes": fit["amplitudes"],
                "frequencies": fit["frequencies"],
                "frequency_errors": fit["frequency_errors"],
                "phases": fit["phases"],
                "t2_star": fit["t2_star"],
                "t2_star_error": fit["t2_star_error"],
                "r2": fit["r2"],
            },
            "fh_frequency": {
                "frequencies": fh_frequency_fit["frequencies"],
                "errors": fh_frequency_fit["errors"],
                "mean": fh_frequency_fit["mean"],
                "mean_error": fh_frequency_fit["mean_error"],
                "components": fh_frequency_fit["components"],
            },
            "fh_x90": {
                "duration": duration,
                "amplitude": fh_x90_pulses[target].amplitude,
                "tau": ramptime,
            },
        }
        figures[target] = fig
        figures[f"{target}_spectrum"] = spectrum_fig
        figures[f"{target}_filtered"] = filtered_fig

    return Result(data=data, figures=figures)


def ef_ramsey_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    resonant_frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    omega_rabis: dict[str, float] | None = None,
    resonant_frequency: float | None = None,
    amplitude: float | None = None,
    omega_rabi: float | None = None,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    zero_fill_factor: int | None = None,
    dc_suppression_width: float | None = None,
    peak_distance: float | None = None,
    n_peaks: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Measure EF Ramsey data and project the IQ trace onto its PCA axis.

    The pulse sequence is
    ``X_ge -> EF X90 -> wait T -> EF X90 -> X_ge -> readout``.
    """
    target_labels = _normalize_ef_targets(exp, targets)
    if detuning is None:
        detuning = 0.001
    if duration is None:
        duration = HPI_DURATION
    if ramptime is None:
        ramptime = HPI_RAMPTIME
    if zero_fill_factor is None:
        zero_fill_factor = 8
    if dc_suppression_width is None:
        dc_suppression_width = 0.00025
    if peak_distance is None:
        peak_distance = 0.00005
    if n_peaks is None:
        n_peaks = 2
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if time_range is None:
        time_range = np.arange(0, 20001, 200)

    sampling_period = exp.ctx.measurement.sampling_period
    time_values = exp.ctx.util.discretize_time_range(
        np.asarray(time_range, dtype=np.float64),
        sampling_period=sampling_period,
    )
    if time_values is None:
        raise ValueError("time_range could not be discretized.")

    normalized_frequencies = _normalize_target_values(
        exp,
        target_labels,
        values=resonant_frequencies,
        scalar_value=resonant_frequency,
        value_name="resonant_frequency",
    )
    normalized_amplitudes = _normalize_target_values(
        exp,
        target_labels,
        values=amplitudes,
        scalar_value=amplitude,
        value_name="amplitude",
    )
    normalized_omega_rabis = _normalize_target_values(
        exp,
        target_labels,
        values=omega_rabis,
        scalar_value=omega_rabi,
        value_name="omega_rabi",
    )

    ef_x90_pulses = {
        target: _build_transition_x90_pulse(
            amplitude=normalized_amplitudes[target],
            omega_rabi=normalized_omega_rabis[target],
            duration=duration,
            ramptime=ramptime,
            sampling_period=sampling_period,
        )
        for target in target_labels
    }
    drive_frequencies = {
        target: normalized_frequencies[target] + detuning for target in target_labels
    }

    data = {}
    figures = {}
    for target in target_labels:
        qubit = exp.ctx.resolve_qubit_label(target)
        ge_label = exp.ctx.resolve_ge_label(target)
        ef_label = exp.ctx.resolve_ef_label(target)
        ef_x90 = ef_x90_pulses[target]

        def ramsey_sequence(
            wait_time: int,
            ge_label: str = ge_label,
            ef_label: str = ef_label,
            ef_x90: FlatTop = ef_x90,
        ) -> PulseSchedule:
            with PulseSchedule() as ps:
                ps.add(ge_label, exp.pulse.x180(ge_label))
                ps.barrier()
                ps.add(ef_label, ef_x90)
                ps.add(ef_label, Blank(wait_time))
                ps.add(ef_label, ef_x90)
                ps.barrier()
                ps.add(ge_label, exp.pulse.x180(ge_label))
            return ps

        with exp.ctx.util.no_output():
            sweep_result = exp.measurement_service.sweep_parameter(
                sequence=ramsey_sequence,
                sweep_range=time_values,
                frequencies={target: drive_frequencies[target]},
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=plot,
            )

        iq = np.asarray(sweep_result.data[qubit].data, dtype=np.complex128)
        projection = _project_iq_pca(iq)
        spectrum = _calculate_spectrum(
            time_range=time_values,
            projected=projection["projected"],
            zero_fill_factor=zero_fill_factor,
            dc_suppression_width=dc_suppression_width,
        )
        peaks = _find_ramsey_spectrum_peaks(
            fft_frequencies=spectrum["fft_frequencies"],
            fft_values=spectrum["fft_values_dc_suppressed"],
            detuning=detuning,
            dc_suppression_width=dc_suppression_width,
            peak_distance=peak_distance,
            n_peaks=n_peaks,
        )
        fit = _fit_projected_ramsey(
            time_range=time_values,
            projected=spectrum["dc_suppressed_waveform"],
            peak_frequencies=peaks["frequencies"],
            n_peaks=n_peaks,
            detuning=detuning,
        )
        ef_frequency_fit = _calculate_transition_frequency_fit(
            drive_frequency=drive_frequencies[target],
            fit_frequencies=fit["frequencies"],
            fit_frequency_errors=fit["frequency_errors"],
        )
        fig = _make_raw_ramsey_figure(
            target=target,
            time_range=time_values,
            projected=projection["projected"],
            resonant_frequency=normalized_frequencies[target],
            drive_frequency=drive_frequencies[target],
            amplitude=ef_x90_pulses[target].amplitude,
            transition_label="EF",
        )
        filtered_fig = _make_filtered_ramsey_figure(
            target=target,
            time_range=time_values,
            filtered_waveform=spectrum["dc_suppressed_waveform"],
            fit_time_range=fit["fit_time_range"],
            fit_curve=fit["fit_curve"],
            transition_label="EF",
        )
        spectrum_fig = _make_spectrum_figure(
            target=target,
            fft_frequencies=spectrum["fft_frequencies_shifted"],
            fft_values=spectrum["fft_values_shifted"],
            peak_frequencies=peaks["frequencies"],
            peak_values=spectrum["fft_values"][peaks["indices"]],
            transition_label="EF",
        )

        if plot:
            _print_ramsey_fit_result(
                target=target,
                fit=fit,
                transition_frequency_fit=ef_frequency_fit,
                transition_label="EF",
            )
            fig.show()
            spectrum_fig.show()
            filtered_fig.show()
        if save_image:
            viz.save_figure(
                fig,
                name=f"ef_ramsey_{target}",
                width=600,
                height=400,
            )
            viz.save_figure(
                filtered_fig,
                name=f"ef_ramsey_filtered_{target}",
                width=600,
                height=400,
            )
            viz.save_figure(
                spectrum_fig,
                name=f"ef_ramsey_spectrum_{target}",
                width=600,
                height=400,
            )

        data[target] = {
            "target": target,
            "qubit": qubit,
            "inputs": {
                "resonant_frequency": normalized_frequencies[target],
                "drive_frequency": drive_frequencies[target],
                "detuning": detuning,
                "chevron_amplitude": normalized_amplitudes[target],
                "omega_rabi": normalized_omega_rabis[target],
                "n_shots": n_shots,
                "shot_interval": shot_interval,
            },
            "traces": {
                "time_range": time_values,
                "iq": iq,
                "projected": projection["projected"],
                "dc_suppressed_waveform": spectrum["dc_suppressed_waveform"],
            },
            "pca": {
                "axis": projection["axis"],
                "center": projection["center"],
            },
            "spectrum": {
                "dc_suppression_width": dc_suppression_width,
                "dc_suppression_alpha": spectrum["dc_suppression_alpha"],
                "peak_distance": peak_distance,
                "peak_frequencies": peaks["frequencies"],
                "peak_amplitudes": peaks["amplitudes"],
            },
            "fit": {
                "status": fit["status"],
                "params": fit["params"],
                "time_range": fit["fit_time_range"],
                "curve": fit["fit_curve"],
                "amplitudes": fit["amplitudes"],
                "frequencies": fit["frequencies"],
                "frequency_errors": fit["frequency_errors"],
                "phases": fit["phases"],
                "t2_star": fit["t2_star"],
                "t2_star_error": fit["t2_star_error"],
                "r2": fit["r2"],
            },
            "ef_frequency": {
                "frequencies": ef_frequency_fit["frequencies"],
                "errors": ef_frequency_fit["errors"],
                "mean": ef_frequency_fit["mean"],
                "mean_error": ef_frequency_fit["mean_error"],
                "components": ef_frequency_fit["components"],
            },
            "ef_x90": {
                "duration": duration,
                "amplitude": ef_x90_pulses[target].amplitude,
                "tau": ramptime,
            },
        }
        figures[target] = fig
        figures[f"{target}_spectrum"] = spectrum_fig
        figures[f"{target}_filtered"] = filtered_fig

    return Result(data=data, figures=figures)


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        target_labels = list(exp.ctx.qubit_labels)
    elif isinstance(targets, str):
        target_labels = [targets]
    else:
        target_labels = list(targets)
    return [exp.ctx.resolve_fh_label(target) for target in target_labels]


def _normalize_ef_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        target_labels = list(exp.ctx.qubit_labels)
    elif isinstance(targets, str):
        target_labels = [targets]
    else:
        target_labels = list(targets)
    return [exp.ctx.resolve_ef_label(target) for target in target_labels]


def _normalize_target_values(
    exp: Experiment,
    targets: Collection[str],
    *,
    values: dict[str, float] | None,
    scalar_value: float | None,
    value_name: str,
) -> dict[str, float]:
    if values is None and scalar_value is None:
        raise ValueError(f"{value_name} must be provided.")
    if len(targets) != 1 and scalar_value is not None:
        raise ValueError(f"Scalar {value_name} is only allowed for a single target.")

    normalized = {}
    for target in targets:
        qubit = exp.ctx.resolve_qubit_label(target)
        if values is None:
            if scalar_value is None:
                raise ValueError(f"{value_name} must be provided for {target}.")
            normalized[target] = float(scalar_value)
        elif target in values:
            normalized[target] = float(values[target])
        elif qubit in values:
            normalized[target] = float(values[qubit])
        else:
            raise ValueError(f"{value_name} must be provided for {target}.")
    return normalized


def _build_transition_x90_pulse(
    *,
    amplitude: float,
    omega_rabi: float,
    duration: float,
    ramptime: float,
    sampling_period: float,
) -> FlatTop:
    if omega_rabi <= 0:
        raise ValueError("omega_rabi must be positive.")
    unit_area = duration - ramptime
    if unit_area <= 0:
        raise ValueError("duration - ramptime must be positive.")
    target_area = 0.25 * amplitude / omega_rabi
    pulse_amplitude = target_area / unit_area
    return FlatTop(
        duration=duration,
        amplitude=pulse_amplitude,
        tau=ramptime,
        sampling_period=sampling_period,
    )


def _project_iq_pca(iq: ArrayLike) -> dict[str, np.ndarray]:
    values = np.asarray(iq, dtype=np.complex128)
    center = np.array([np.mean(values.real), np.mean(values.imag)])
    centered = np.column_stack([values.real - center[0], values.imag - center[1]])
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = np.asarray(vh[0], dtype=np.float64)
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0])
    projected = np.asarray(centered @ axis, dtype=np.float64)
    if projected[0] < 0:
        axis = -axis
        projected = -projected
    return {
        "projected": projected,
        "axis": axis,
        "center": center,
    }


def _calculate_spectrum(
    *,
    time_range: ArrayLike,
    projected: ArrayLike,
    zero_fill_factor: int,
    dc_suppression_width: float,
) -> _SpectrumResult:
    time_values = np.asarray(time_range, dtype=np.float64)
    projected_values = np.asarray(projected, dtype=np.float64)
    if time_values.ndim != 1 or projected_values.ndim != 1:
        raise ValueError("time_range and projected must be one-dimensional.")
    if time_values.size != projected_values.size:
        raise ValueError("time_range and projected must have the same length.")
    if time_values.size < 2:
        raise ValueError("At least two points are required for FFT.")
    if zero_fill_factor < 1:
        raise ValueError("zero_fill_factor must be positive.")
    if dc_suppression_width < 0:
        raise ValueError("dc_suppression_width must be non-negative.")

    dt_values = np.diff(time_values)
    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError("time_range must be evenly spaced for FFT.")

    padded_length = _next_power_of_two(time_values.size * zero_fill_factor)
    zero_filled = np.zeros(padded_length, dtype=np.complex128)
    zero_filled[: projected_values.size] = projected_values.astype(np.complex128)

    fft_values = np.fft.fft(zero_filled)
    fft_frequencies = np.fft.fftfreq(padded_length, d=float(dt_values[0]))
    dc_suppression_window = _make_dc_tukey_suppression_window(
        fft_frequencies=fft_frequencies,
        dc_suppression_width=dc_suppression_width,
        alpha=0.2,
    )
    fft_values_dc_suppressed = fft_values * dc_suppression_window
    dc_suppressed_zero_filled = np.fft.ifft(fft_values_dc_suppressed)
    return {
        "fft_frequencies": fft_frequencies,
        "fft_values": fft_values,
        "dc_suppression_alpha": 0.2,
        "fft_values_dc_suppressed": fft_values_dc_suppressed,
        "fft_frequencies_shifted": np.fft.fftshift(fft_frequencies),
        "fft_values_shifted": np.fft.fftshift(fft_values),
        "dc_suppressed_waveform": dc_suppressed_zero_filled[
            : projected_values.size
        ].real,
    }


def _make_dc_tukey_suppression_window(
    *,
    fft_frequencies: ArrayLike,
    dc_suppression_width: float,
    alpha: float,
) -> np.ndarray:
    frequency_values = np.asarray(fft_frequencies, dtype=np.float64)
    if dc_suppression_width == 0:
        return np.ones_like(frequency_values, dtype=np.float64)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")

    abs_frequencies = np.abs(frequency_values)
    lowpass = np.zeros_like(frequency_values, dtype=np.float64)
    plateau_edge = (1 - alpha) * dc_suppression_width
    plateau_mask = abs_frequencies <= plateau_edge
    taper_mask = (plateau_edge < abs_frequencies) & (
        abs_frequencies < dc_suppression_width
    )
    lowpass[plateau_mask] = 1.0
    if np.any(taper_mask):
        taper_position = (abs_frequencies[taper_mask] - plateau_edge) / (
            alpha * dc_suppression_width
        )
        lowpass[taper_mask] = 0.5 * (1 + np.cos(np.pi * taper_position))
    return 1 - lowpass


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _find_ramsey_spectrum_peaks(
    *,
    fft_frequencies: ArrayLike,
    fft_values: ArrayLike,
    detuning: float,
    dc_suppression_width: float,
    peak_distance: float | None,
    n_peaks: int,
) -> dict[str, np.ndarray]:
    frequency_values = np.asarray(fft_frequencies, dtype=np.float64)
    spectrum_values = np.asarray(fft_values, dtype=np.complex128)
    if frequency_values.shape != spectrum_values.shape:
        raise ValueError("fft_frequencies and fft_values must have the same shape.")
    if dc_suppression_width < 0:
        raise ValueError("dc_suppression_width must be non-negative.")
    if n_peaks < 1:
        raise ValueError("n_peaks must be positive.")

    if detuning > 0:
        mask = frequency_values < -dc_suppression_width
    elif detuning < 0:
        mask = frequency_values > dc_suppression_width
    else:
        mask = np.abs(frequency_values) > dc_suppression_width

    masked_indices = np.flatnonzero(mask)
    if masked_indices.size == 0:
        return _empty_peak_result()

    masked_frequencies = frequency_values[masked_indices]
    masked_amplitudes = np.abs(spectrum_values[masked_indices])
    sort_order = np.argsort(masked_frequencies)
    sorted_indices = masked_indices[sort_order]
    sorted_frequencies = masked_frequencies[sort_order]
    sorted_amplitudes = masked_amplitudes[sort_order]

    find_peaks_kwargs = {}
    if peak_distance is not None:
        if peak_distance <= 0:
            raise ValueError("peak_distance must be positive.")
        if sorted_frequencies.size > 1:
            df = float(np.median(np.diff(sorted_frequencies)))
            if df > 0:
                find_peaks_kwargs["distance"] = max(1, int(np.ceil(peak_distance / df)))

    peak_positions, _ = find_peaks(sorted_amplitudes, **find_peaks_kwargs)
    if peak_positions.size == 0:
        return _empty_peak_result()

    peak_amplitudes = sorted_amplitudes[peak_positions]
    strongest_positions = peak_positions[np.argsort(peak_amplitudes)[::-1][:n_peaks]]
    strongest_positions = strongest_positions[
        np.argsort(sorted_frequencies[strongest_positions])
    ]
    indices = sorted_indices[strongest_positions]

    return {
        "indices": indices,
        "frequencies": frequency_values[indices],
        "values": spectrum_values[indices],
        "amplitudes": np.abs(spectrum_values[indices]),
    }


def _empty_peak_result() -> dict[str, np.ndarray]:
    return {
        "indices": np.array([], dtype=int),
        "frequencies": np.array([], dtype=np.float64),
        "values": np.array([], dtype=np.complex128),
        "amplitudes": np.array([], dtype=np.float64),
    }


def _fit_projected_ramsey(
    *,
    time_range: ArrayLike,
    projected: ArrayLike,
    peak_frequencies: ArrayLike,
    n_peaks: int,
    detuning: float,
) -> _RamseyFitResult:
    time_values = np.asarray(time_range, dtype=np.float64)
    projected_values = np.asarray(projected, dtype=np.float64)
    peak_frequency_values = np.asarray(peak_frequencies, dtype=np.float64)
    if peak_frequency_values.size == 0:
        return _empty_fit_result(time_values, projected_values)

    fit_frequencies = _signed_fit_frequencies(
        peak_frequencies=peak_frequency_values[:n_peaks],
        detuning=detuning,
    )
    amplitude0 = projected_values[0] / fit_frequencies.size
    time_span = float(np.max(time_values) - np.min(time_values))
    t2_star0 = time_span if time_span > 0 else 1.0

    initial_params = []
    lower_bounds = []
    upper_bounds = []
    for frequency in fit_frequencies:
        initial_params.extend([amplitude0, frequency, 0.0])
        if frequency < 0:
            lower_bounds.extend([-np.inf, -np.inf, -2 * np.pi])
            upper_bounds.extend([np.inf, 0.0, 2 * np.pi])
        elif frequency > 0:
            lower_bounds.extend([-np.inf, 0.0, -2 * np.pi])
            upper_bounds.extend([np.inf, np.inf, 2 * np.pi])
        else:
            lower_bounds.extend([-np.inf, -np.inf, -2 * np.pi])
            upper_bounds.extend([np.inf, np.inf, 2 * np.pi])
    initial_params.append(t2_star0)
    lower_bounds.append(np.finfo(np.float64).eps)
    upper_bounds.append(np.inf)
    smooth_time_values = np.linspace(
        float(time_values[0]),
        float(time_values[-1]),
        max(1000, time_values.size * 10),
    )

    try:
        params, covariance = curve_fit(
            _multi_decaying_cosine,
            time_values,
            projected_values,
            p0=np.asarray(initial_params, dtype=np.float64),
            bounds=(
                np.asarray(lower_bounds, dtype=np.float64),
                np.asarray(upper_bounds, dtype=np.float64),
            ),
            maxfev=10000,
        )
        fit_curve_at_data = _multi_decaying_cosine(
            time_values,
            *params,
        )
        fit_curve = _multi_decaying_cosine(
            smooth_time_values,
            *params,
        )
        residual = projected_values - fit_curve_at_data
        denom = np.sum((projected_values - np.mean(projected_values)) ** 2)
        r2 = np.nan if denom == 0 else 1 - np.sum(residual**2) / denom
        parameter_errors = _parameter_errors_from_covariance(covariance)
        if parameter_errors.size != params.size:
            parameter_errors = np.full_like(params, np.nan, dtype=np.float64)
        status = "success"
    except (RuntimeError, ValueError):
        params = np.asarray(initial_params, dtype=np.float64)
        parameter_errors = np.full_like(params, np.nan, dtype=np.float64)
        fit_curve_at_data = _multi_decaying_cosine(
            time_values,
            *params,
        )
        fit_curve = _multi_decaying_cosine(
            smooth_time_values,
            *params,
        )
        r2 = np.nan
        status = "error"

    return {
        "status": status,
        "params": params,
        "fit_time_range": smooth_time_values,
        "fit_curve": fit_curve,
        "amplitudes": params[:-1][0::3],
        "frequencies": params[:-1][1::3],
        "frequency_errors": parameter_errors[:-1][1::3],
        "phases": params[:-1][2::3],
        "t2_star": float(params[-1]),
        "t2_star_error": float(parameter_errors[-1]),
        "r2": float(r2),
    }


def _parameter_errors_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance_values = np.asarray(covariance, dtype=np.float64)
    if (
        covariance_values.ndim != 2
        or covariance_values.shape[0] != covariance_values.shape[1]
    ):
        return np.array([], dtype=np.float64)
    diagonal = np.diag(covariance_values)
    return np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))


def _calculate_transition_frequency_fit(
    *,
    drive_frequency: float,
    fit_frequencies: ArrayLike,
    fit_frequency_errors: ArrayLike,
) -> _TransitionFrequencyFit:
    ramsey_frequencies = np.asarray(fit_frequencies, dtype=np.float64)
    ramsey_frequency_errors = np.asarray(fit_frequency_errors, dtype=np.float64)
    if ramsey_frequencies.size == 0:
        return {
            "frequencies": np.array([], dtype=np.float64),
            "errors": np.array([], dtype=np.float64),
            "mean": np.nan,
            "mean_error": np.nan,
            "components": {},
        }
    if ramsey_frequency_errors.size != ramsey_frequencies.size:
        ramsey_frequency_errors = np.full_like(ramsey_frequencies, np.nan)

    transition_frequencies = float(drive_frequency) + ramsey_frequencies
    order = np.argsort(transition_frequencies)
    sorted_frequencies = transition_frequencies[order]
    sorted_errors = ramsey_frequency_errors[order]
    return {
        "frequencies": sorted_frequencies,
        "errors": sorted_errors,
        "mean": float(np.mean(sorted_frequencies)),
        "mean_error": _calculate_mean_error(sorted_errors),
        "components": _make_frequency_components(sorted_frequencies, sorted_errors),
    }


def _make_frequency_components(
    frequencies: ArrayLike,
    errors: ArrayLike,
) -> dict[str, dict[str, float]]:
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    error_values = np.asarray(errors, dtype=np.float64)
    if error_values.size != frequency_values.size:
        error_values = np.full_like(frequency_values, np.nan)
    return {
        str(index): {
            "frequency": float(frequency),
            "error": float(error),
        }
        for index, (frequency, error) in enumerate(
            zip(frequency_values, error_values, strict=True),
            start=1,
        )
    }


def _calculate_mean_error(errors: ArrayLike) -> float:
    error_values = np.asarray(errors, dtype=np.float64)
    if error_values.size == 0 or not np.all(np.isfinite(error_values)):
        return np.nan
    return float(np.sqrt(np.sum(error_values**2)) / error_values.size)


def _signed_fit_frequencies(
    *,
    peak_frequencies: ArrayLike,
    detuning: float,
) -> np.ndarray:
    peak_frequency_values = np.asarray(peak_frequencies, dtype=np.float64)
    if detuning > 0:
        return -np.abs(peak_frequency_values)
    if detuning < 0:
        return np.abs(peak_frequency_values)
    return peak_frequency_values


def _empty_fit_result(
    time_values: np.ndarray,
    projected_values: np.ndarray,
) -> _RamseyFitResult:
    return {
        "status": "no_peaks",
        "params": np.array([], dtype=np.float64),
        "fit_time_range": time_values,
        "fit_curve": np.full_like(projected_values, np.nan, dtype=np.float64),
        "amplitudes": np.array([], dtype=np.float64),
        "frequencies": np.array([], dtype=np.float64),
        "frequency_errors": np.array([], dtype=np.float64),
        "phases": np.array([], dtype=np.float64),
        "t2_star": np.nan,
        "t2_star_error": np.nan,
        "r2": np.nan,
    }


def _print_ramsey_fit_result(
    *,
    target: str,
    fit: _RamseyFitResult,
    transition_frequency_fit: _TransitionFrequencyFit,
    transition_label: str = "FH",
) -> None:
    amplitudes = np.asarray(fit["amplitudes"], dtype=np.float64)
    frequencies = np.asarray(fit["frequencies"], dtype=np.float64)
    frequency_errors = np.asarray(fit["frequency_errors"], dtype=np.float64)
    phases = np.asarray(fit["phases"], dtype=np.float64)
    transition_lower = transition_label.lower()
    print(f"{transition_label} Ramsey fit : {target}")
    print(f"  status = {fit['status']}")
    for index, (amplitude, frequency, frequency_error, phase) in enumerate(
        zip(amplitudes, frequencies, frequency_errors, phases, strict=True),
        start=1,
    ):
        print(
            f"  component {index}: "
            f"a={amplitude:.6g}, "
            f"f={_format_value_with_error(frequency * 1e3, frequency_error * 1e3)} MHz, "
            f"phi={phase:.6g} rad",
        )
    _print_transition_frequency_fit(
        transition_lower=transition_lower,
        frequency_fit=transition_frequency_fit,
    )
    print(
        f"  T2* = {_format_value_with_error(fit['t2_star'], fit['t2_star_error'])} ns",
    )
    print(f"  R^2 = {float(fit['r2']):.6g}")


def _print_transition_frequency_fit(
    *,
    transition_lower: str,
    frequency_fit: _TransitionFrequencyFit,
) -> None:
    frequencies = np.asarray(frequency_fit["frequencies"], dtype=np.float64)
    errors = np.asarray(frequency_fit["errors"], dtype=np.float64)
    if errors.size != frequencies.size:
        errors = np.full_like(frequencies, np.nan)
    for index, (frequency, error) in enumerate(
        zip(frequencies, errors, strict=True),
        start=1,
    ):
        print(
            f"  {transition_lower}_freq_{index} = "
            f"{_format_value_with_error(frequency, error)} GHz",
        )
    print(
        f"  {transition_lower}_freq_mean = "
        f"{_format_value_with_error(frequency_fit['mean'], frequency_fit['mean_error'])} GHz",
    )


def _format_value_with_error(value: float, error: float) -> str:
    value_float = float(value)
    error_float = float(error)
    if np.isfinite(value_float) and np.isfinite(error_float):
        return f"{value_float:.9g} +/- {error_float:.3g}"
    return f"{value_float:.9g} +/- nan"


def _multi_decaying_cosine(
    time_values: np.ndarray,
    *params: float,
) -> np.ndarray:
    values = np.zeros_like(time_values, dtype=np.float64)
    cosine_params = params[:-1]
    t2_star = params[-1]
    decay = np.exp(-time_values / t2_star)
    for amplitude, frequency, phase in np.asarray(cosine_params).reshape(-1, 3):
        values += (
            amplitude * decay * np.cos(2 * np.pi * frequency * time_values + phase)
        )
    return values


def _make_raw_ramsey_figure(
    *,
    target: str,
    time_range: ArrayLike,
    projected: ArrayLike,
    resonant_frequency: float,
    drive_frequency: float,
    amplitude: float,
    transition_label: str = "FH",
) -> go.Figure:
    time_values = np.asarray(time_range, dtype=np.float64)
    projected_values = np.asarray(projected, dtype=np.float64)

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=projected_values,
            mode="markers+lines",
            name="PCA projection",
        ),
    )
    fig.update_layout(
        title=(
            f"{transition_label} Ramsey raw : {target}<br>"
            f"<sup>f_res={resonant_frequency:.6f} GHz, "
            f"f_drive={drive_frequency:.6f} GHz, "
            f"x90_amp={amplitude:.6g}</sup>"
        ),
        xaxis_title="Wait time (ns)",
        yaxis_title="Signal (arb. units)",
        width=600,
        height=400,
    )
    return fig


def _make_filtered_ramsey_figure(
    *,
    target: str,
    time_range: ArrayLike,
    filtered_waveform: ArrayLike,
    fit_time_range: ArrayLike,
    fit_curve: ArrayLike,
    transition_label: str = "FH",
) -> go.Figure:
    time_values = np.asarray(time_range, dtype=np.float64)
    filtered_values = np.asarray(filtered_waveform, dtype=np.float64)
    fit_time_values = np.asarray(fit_time_range, dtype=np.float64)
    fit_values = np.asarray(fit_curve, dtype=np.float64)

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=filtered_values,
            mode="markers+lines",
            name="DC-suppressed waveform",
        ),
    )
    if fit_values.size == fit_time_values.size and np.any(np.isfinite(fit_values)):
        fig.add_trace(
            go.Scatter(
                x=fit_time_values,
                y=fit_values,
                mode="lines",
                name="Fit",
            ),
        )
    fig.update_layout(
        title=f"{transition_label} Ramsey DC-suppressed fit : {target}",
        xaxis_title="Wait time (ns)",
        yaxis_title="Signal (arb. units)",
        width=600,
        height=400,
    )
    return fig


def _make_spectrum_figure(
    *,
    target: str,
    fft_frequencies: ArrayLike,
    fft_values: ArrayLike,
    peak_frequencies: ArrayLike,
    peak_values: ArrayLike,
    transition_label: str = "FH",
) -> go.Figure:
    frequency_values = np.asarray(fft_frequencies, dtype=np.float64)
    spectrum_values = np.asarray(fft_values, dtype=np.complex128)
    peak_frequency_values = np.asarray(peak_frequencies, dtype=np.float64)
    peak_spectrum_values = np.asarray(peak_values, dtype=np.complex128)

    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            x=frequency_values * 1e3,
            y=spectrum_values.real,
            mode="lines",
            name="Real",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frequency_values * 1e3,
            y=spectrum_values.imag,
            mode="lines",
            name="Imag",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frequency_values * 1e3,
            y=np.abs(spectrum_values),
            mode="lines",
            line=dict(dash="dot"),
            name="Abs",
        ),
    )
    if peak_frequency_values.size > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_frequency_values * 1e3,
                y=np.abs(peak_spectrum_values),
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name="Picked peaks",
            ),
        )
    fig.update_layout(
        title=f"{transition_label} Ramsey spectrum : {target}",
        xaxis_title="Frequency (MHz)",
        yaxis_title="FFT value",
        width=600,
        height=400,
    )
    return fig
