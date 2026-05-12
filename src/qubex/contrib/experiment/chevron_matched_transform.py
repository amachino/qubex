"""Chevron-based qubit frequency calibration using matched-transform analysis."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from qxpulse import Rect
from tqdm import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result


def estimate_qubit_frequency_from_chevron(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    detuning_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    omega_rabi_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    quadratic_window: int | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Estimate qubit resonance frequencies from chevron measurements.

    This function measures chevron patterns using fixed drive frequencies
    and amplitudes, and estimates the qubit resonance frequency
    (ω_q) and resonant Rabi frequency (ω_Rabi) using matched-transform
    analysis.

    Unlike estimate_qubit_frequency_from_chevron_adaptive(),
    this function does not perform rough-search measurements or adaptive
    updates of the drive frequency/amplitude.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context.
    targets : Collection[str] or str, optional
        Target qubit label(s).
        If None, all available qubits are used.
    detuning_range : array-like, optional
        Frequency detuning range (GHz) for chevron measurements.
        Default: np.linspace(-0.1, 0.1, 51)
    time_range : array-like, optional
        Pulse duration range (ns).
        Default: DEFAULT_RABI_TIME_RANGE
    frequencies : dict[str, float], optional
        Drive frequencies (GHz) for each target qubit.
        If None, calibrated qubit frequencies are used.
    amplitudes : dict[str, float], optional
        Drive amplitudes for each target qubit.
        If None, default control amplitudes are used.
    omega_rabi_range : array-like, optional
        Trial resonant Rabi frequencies (GHz) used in the matched transform.
    n_shots : int, optional
        Number of shots.
        Default: DEFAULT_SHOTS // 4
    shot_interval : float, optional
        Interval between shots.
    quadratic_window : int, optional
        Half-width of the local quadratic fitting window used for
        sub-grid peak refinement.
        quadratic_window=2 uses a 5x5 patch.
    plot : bool, optional
        If True, display generated figures.
    save_image : bool, optional
        If True, save generated figures.

    Returns
    -------
    Result
        data:
            {
                "results": {
                    target: {
                        "omega_q":
                            estimated qubit frequency (GHz),
                        "omega_rabi":
                            estimated resonant Rabi frequency (GHz),
                        "peak_prominence_ratio":
                            confidence metric defined as the ratio between
                            the main transform peak and the largest competing
                            peak outside a neighborhood of the main peak,
                        "frequency_used":
                            drive frequency used for measurement (GHz),
                        "amplitude_used":
                            drive amplitude used for measurement,
                        "chevron_data":
                            projected chevron data,
                        "transform":
                            matched-transform map,
                    }
                },
                "time_range":
                    pulse duration axis (ns),
                "detuning_range":
                    detuning axis (GHz),
                "resonant_frequencies":
                    estimated qubit frequencies,
                "omega_rabis":
                    estimated resonant Rabi frequencies,
                "amplitudes_used":
                    amplitudes used for each target,
                "peak_prominence_ratios":
                    peak prominence ratios for each target,
            }
        figures:
            {
                f"{target}_measurement":
                    chevron measurement figure,
                f"{target}_transform":
                    matched-transform analysis figure,
            }

    Notes
    -----
    - Chevron data are projected onto a global PCA axis in the IQ plane
        before analysis.
    - The matched transform integrates over both drive frequency and
        pulse duration and supports nonuniform sampling grids.
    - The overall sign ambiguity of the transform is fixed such that
        the dominant peak becomes positive.
    - Quadratic refinement improves sub-grid peak estimation accuracy.
    - The peak prominence ratio serves as a confidence metric for
        peak detection.
    """
    if targets is None:
        targets = exp.ctx.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if detuning_range is None:
        detuning_range = np.linspace(-0.1, 0.1, 51)
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if omega_rabi_range is None:
        omega_rabi_range = 0.1 * (10 ** np.linspace(0, 1, 256) - 1) / 9
    if n_shots is None:
        n_shots = max(1, DEFAULT_SHOTS // 4)
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if quadratic_window is None:
        quadratic_window = 2
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    detuning_range = np.asarray(detuning_range, dtype=float)
    time_range = np.asarray(time_range, dtype=float)
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)

    if frequencies is None:
        frequencies = {target: exp.ctx.targets[target].frequency for target in targets}

    if amplitudes is None:
        amplitudes = {
            target: exp.ctx.params.control_amplitude[target] for target in targets
        }

    results_data = {}
    resonant_frequencies: dict[str, float] = {}
    omega_rabis: dict[str, float] = {}
    peak_prominence_ratios: dict[str, float] = {}
    figures = {}

    for target in targets:
        print(f"=== Target: {target} ===")

        measurement_result, analysis_result = _measure_and_analyze_chevron(
            exp,
            target,
            detuning_range=detuning_range,
            time_range=time_range,
            frequency=frequencies[target],
            amplitude=amplitudes[target],
            omega_rabi_range=omega_rabi_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            refine_peak_quadratic=True,
            quadratic_window=quadratic_window,
            plot=plot,
            save_image=save_image,
        )

        analysis = analysis_result.data

        omega_q = analysis["omega_q"]
        omega_rabi = analysis["omega_rabi"]
        peak_prominence_ratio = analysis["peak_prominence_ratio"]

        print(
            f"[RESULT] ω_q={omega_q:.6f} GHz,"
            f" ω_Rabi={omega_rabi:.6f} GHz"
            f" peak_prominence_ratio={peak_prominence_ratio:.3f}"
        )

        results_data[target] = {
            "omega_q": omega_q,
            "omega_rabi": omega_rabi,
            "peak_prominence_ratio": peak_prominence_ratio,
            "frequency_used": frequencies[target],
            "amplitude_used": amplitudes[target],
            "chevron_data": measurement_result.data["chevron_data"],
            "transform": analysis["transform"],
        }

        resonant_frequencies[target] = omega_q
        omega_rabis[target] = omega_rabi
        peak_prominence_ratios[target] = peak_prominence_ratio
        figures[f"{target}_measurement"] = measurement_result.figure
        figures[f"{target}_transform"] = analysis_result.figure

    return Result(
        data={
            "results": results_data,
            "time_range": time_range,
            "detuning_range": detuning_range,
            "resonant_frequencies": resonant_frequencies,
            "omega_rabis": omega_rabis,
            "amplitudes_used": amplitudes,
            "peak_prominence_ratios": peak_prominence_ratios,
        },
        figures=figures,
    )


def estimate_qubit_frequency_from_chevron_adaptive(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    detuning_range: ArrayLike | None = None,
    detuning_range_rough_search: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    omega_rabi_range: ArrayLike | None = None,
    target_omega_rabi: float | None = None,
    n_shots: int | None = None,
    n_shots_rough_search: int | None = None,
    shot_interval: float | None = None,
    quadratic_window: int | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    enable_rough_search: bool | None = None,
) -> Result:
    """
    Adaptively estimate qubit resonance frequencies from chevron measurements.

    This function performs adaptive chevron-based estimation of the qubit
    resonance frequency (ω_q) and resonant Rabi frequency (ω_Rabi)
    using matched-transform analysis.

    The procedure consists of:

    1. Rough-search stage (optional)
        - Measure a coarse chevron pattern over a wide detuning range.
        - Estimate ω_q and ω_Rabi using matched-transform analysis
            with quadratic peak refinement.
        - Update:
            * drive frequency ← estimated ω_q
            * drive amplitude ← rescaled toward target_omega_rabi

    2. Final measurement stage
        - Measure a chevron pattern over a narrower detuning range.
        - Perform matched-transform analysis with quadratic peak refinement
            to obtain final estimates of ω_q and ω_Rabi.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context.
    targets : Collection[str] or str, optional
        Target qubit label(s).
        If None, all available qubits are used.
    detuning_range : array-like, optional
        Detuning range (GHz) for the final measurement.
        Default: np.linspace(-0.05, 0.05, 51)
    detuning_range_rough_search : array-like, optional
        Detuning range (GHz) for the rough-search stage.
        Default: np.linspace(-0.1, 0.1, 51)
    time_range : array-like, optional
        Pulse duration range (ns).
        Default: DEFAULT_RABI_TIME_RANGE
    frequencies : dict[str, float], optional
        Initial drive frequencies (GHz) for each target qubit.
        If None, calibrated qubit frequencies are used.
    amplitudes : dict[str, float], optional
        Initial drive amplitudes for each target qubit.
        If None, default control amplitudes are used.
    omega_rabi_range : array-like, optional
        Trial resonant Rabi frequencies (GHz) used in the matched transform.
    target_omega_rabi : float, optional
        Target resonant Rabi frequency (GHz) used to adaptively rescale
        the drive amplitude.
        Default: 0.0125
    n_shots : int, optional
        Number of shots for the final measurement.
        Default: DEFAULT_SHOTS // 4
    n_shots_rough_search : int, optional
        Number of shots for rough-search measurements.
        Default: n_shots
    shot_interval : float, optional
        Interval between shots.
    quadratic_window : int, optional
        Half-width of the local quadratic fitting window used for
        sub-grid peak refinement.
        quadratic_window=2 uses a 5x5 patch.
    plot : bool, optional
        If True, display generated figures.
    save_image : bool, optional
        If True, save generated figures.
    enable_rough_search : bool, optional
        If True, perform a rough-search measurement before the final stage.

    Returns
    -------
    Result
        data:
            {
                "results": {
                    target: {
                        "omega_q":
                            estimated qubit frequency (GHz),
                        "omega_rabi":
                            estimated resonant Rabi frequency (GHz),
                        "peak_prominence_ratio":
                            confidence metric defined as the ratio between
                            the main transform peak and the largest competing
                            peak outside a neighborhood of the main peak,
                        "frequency_used":
                            final drive frequency used (GHz),
                        "amplitude_used":
                            final drive amplitude used,
                        "chevron_data":
                            projected chevron data,
                        "transform":
                            matched-transform map,
                    }
                },
                "time_range":
                    pulse duration axis (ns),
                "detuning_range":
                    final detuning axis (GHz),
                "resonant_frequencies":
                    estimated qubit frequencies,
                "omega_rabis":
                    estimated resonant Rabi frequencies,
                "amplitudes_used":
                    final amplitudes used for each target,
                "peak_prominence_ratios":
                    peak prominence ratios for each target,
                "rough_search_results":
                    rough-search measurement and analysis results,
            }

        figures:
            {
                f"{target}_measurement":
                    final chevron measurement figure,

                f"{target}_transform":
                    final matched-transform analysis figure,

                f"{target}_rough_measurement":
                    rough-search chevron measurement figure,

                f"{target}_rough_transform":
                    rough-search matched-transform analysis figure,
            }

    Notes
    -----
    - Chevron data are projected onto a global PCA axis in the IQ plane
        before analysis.
    - The matched transform integrates over both drive frequency and
        pulse duration and supports nonuniform sampling grids.
    - The overall sign ambiguity of the transform is fixed such that
        the dominant peak becomes positive.
    - Drive amplitudes are adaptively updated to approach the target
        resonant Rabi frequency.
    - Quadratic refinement is applied during both the rough-search
        and final measurement stages.
    - The peak prominence ratio serves as a confidence metric for
        peak detection.
    """
    if targets is None:
        targets = exp.ctx.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 51)
    if detuning_range_rough_search is None:
        detuning_range_rough_search = np.linspace(-0.1, 0.1, 51)
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if omega_rabi_range is None:
        omega_rabi_range = 0.1 * (10 ** np.linspace(0, 1, 256) - 1) / 9
    if target_omega_rabi is None:
        target_omega_rabi = 0.0125
    if n_shots is None:
        n_shots = max(1, DEFAULT_SHOTS // 4)
    if n_shots_rough_search is None:
        n_shots_rough_search = n_shots
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if quadratic_window is None:
        quadratic_window = 2
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if enable_rough_search is None:
        enable_rough_search = True

    time_range = np.asarray(time_range, dtype=float)
    if time_range.size < 2:
        raise ValueError("time_range must contain at least two points.")
    detuning_range = np.asarray(detuning_range, dtype=float)
    detuning_range_rough_search = np.asarray(detuning_range_rough_search, dtype=float)
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)

    if frequencies is None:
        frequencies = {target: exp.ctx.targets[target].frequency for target in targets}

    if amplitudes is None:
        amplitudes = {
            target: exp.ctx.params.control_amplitude[target] for target in targets
        }

    results_data = {}
    results_data_rough_search = {}
    resonant_frequencies: dict[str, float] = {}
    omega_rabis: dict[str, float] = {}
    amplitudes_used: dict[str, float] = {}
    peak_prominence_ratios: dict[str, float] = {}
    figures = {}

    for target in targets:
        freq = frequencies[target]
        amp = amplitudes[target]

        print(f"=== Target: {target} ===")

        if enable_rough_search:
            print("[rough search]")

            measurement_result, analysis_result = _measure_and_analyze_chevron(
                exp,
                target,
                detuning_range=detuning_range_rough_search,
                time_range=time_range,
                frequency=freq,
                amplitude=amp,
                omega_rabi_range=omega_rabi_range,
                n_shots=n_shots_rough_search,
                shot_interval=shot_interval,
                refine_peak_quadratic=True,
                quadratic_window=quadratic_window,
                plot=plot,
                save_image=save_image,
            )
            analysis = analysis_result.data

            omega_q_est = analysis["omega_q"]
            omega_rabi_est = analysis["omega_rabi"]
            peak_prominence_ratio_rough = analysis["peak_prominence_ratio"]

            print(
                f"[rough result] ω_q={omega_q_est:.6f},"
                f" ω_Rabi={omega_rabi_est:.6f},"
                f" peak_prominence_ratio={peak_prominence_ratio_rough:.3f}"
            )

            freq_old = freq
            amp_old = amp

            freq = omega_q_est

            scale = target_omega_rabi / max(omega_rabi_est, 1e-12)
            scale = np.clip(scale, 0.1, 10.0)
            amp *= scale

            print(
                f"[update] freq: {freq_old:.6f} → {freq:.6f}, "
                f"amp: {amp_old:.6g} → {amp:.6g} (scale: {scale:.3f})"
            )

            results_data_rough_search[target] = {
                "omega_q": omega_q_est,
                "omega_rabi": omega_rabi_est,
                "peak_prominence_ratio": peak_prominence_ratio_rough,
                "frequency_used": freq_old,
                "amplitude_used": amp_old,
                "scale": scale,
                "chevron_data": measurement_result.data["chevron_data"],
                "transform": analysis["transform"],
            }

            figures[f"{target}_rough_measurement"] = measurement_result.figure
            figures[f"{target}_rough_transform"] = analysis_result.figure

        print("[final measurement]")

        measurement_result, analysis_result = _measure_and_analyze_chevron(
            exp,
            target,
            detuning_range=detuning_range,
            time_range=time_range,
            frequency=freq,
            amplitude=amp,
            omega_rabi_range=omega_rabi_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            refine_peak_quadratic=True,
            quadratic_window=quadratic_window,
            plot=plot,
            save_image=save_image,
        )
        analysis = analysis_result.data

        omega_q_final = analysis["omega_q"]
        omega_rabi_final = analysis["omega_rabi"]
        peak_prominence_ratio_final = analysis["peak_prominence_ratio"]

        print(
            f"[FINAL] ω_q={omega_q_final:.6f} GHz,"
            f" ω_Rabi={omega_rabi_final:.6f} GHz,"
            f" peak_prominence_ratio={peak_prominence_ratio_final:.3f}"
        )

        results_data[target] = {
            "omega_q": omega_q_final,
            "omega_rabi": omega_rabi_final,
            "peak_prominence_ratio": peak_prominence_ratio_final,
            "frequency_used": freq,
            "amplitude_used": amp,
            "chevron_data": measurement_result.data["chevron_data"],
            "transform": analysis["transform"],
        }

        resonant_frequencies[target] = omega_q_final
        omega_rabis[target] = omega_rabi_final
        peak_prominence_ratios[target] = peak_prominence_ratio_final
        amplitudes_used[target] = amp
        figures[f"{target}_measurement"] = measurement_result.figure
        figures[f"{target}_transform"] = analysis_result.figure

    return Result(
        data={
            "results": results_data,
            "time_range": time_range,
            "detuning_range": detuning_range,
            "resonant_frequencies": resonant_frequencies,
            "omega_rabis": omega_rabis,
            "amplitudes_used": amplitudes_used,
            "peak_prominence_ratios": peak_prominence_ratios,
            "rough_search_results": results_data_rough_search,
        },
        figures=figures,
    )


def _measure_and_analyze_chevron(
    exp: Experiment,
    target: str,
    *,
    detuning_range: ArrayLike,
    time_range: ArrayLike,
    frequency: float,
    amplitude: float,
    omega_rabi_range: ArrayLike,
    n_shots: int,
    shot_interval: float,
    refine_peak_quadratic: bool,
    quadratic_window: int,
    plot: bool,
    save_image: bool,
) -> tuple[Result, Result]:
    """
    Measure a chevron pattern and analyze it using matched transform.

    Returns
    -------
    tuple[Result, Result]
        (measurement_result, analysis_result)
    """
    measurement_result = measure_chevron_pattern(
        exp,
        target,
        detuning_range=detuning_range,
        time_range=time_range,
        frequency=frequency,
        amplitude=amplitude,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        save_image=save_image,
    )

    omega_q_range = np.linspace(
        frequency + 2 * np.min(detuning_range),
        frequency + 2 * np.max(detuning_range),
        1024,
    )

    analysis_result = analyze_chevron_matched_transform(
        measurement_result,
        target,
        omega_q_range=omega_q_range,
        omega_rabi_range=omega_rabi_range,
        refine_peak_quadratic=refine_peak_quadratic,
        quadratic_window=quadratic_window,
        plot=plot,
        save_image=save_image,
    )

    return measurement_result, analysis_result


def measure_chevron_pattern(
    exp: Experiment,
    target: str,
    *,
    detuning_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    frequency: float | None = None,
    amplitude: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Measure a chevron pattern for a single qubit.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context.
    target:
        Target qubit label.
    detuning_range:
        Frequency detuning range (GHz).
    time_range:
        Pulse duration range (ns).
    frequency:
        Base frequency (GHz).
    amplitude:
        Drive amplitude.
    n_shots:
        Number of shots.
    shot_interval:
        Interval between shots.
    plot:
        If True, display heatmap.
    save_image:
        If True, save the generated heatmap figure.

    Returns
    -------
    Result
        data = {
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequency": frequency,
            "amplitude": amplitude,
            "chevron_data":
                projected chevron data with shape
                (len(time_range), len(detuning_range)),
        }
        figure:
            Chevron heatmap figure.
    """
    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 51)
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if frequency is None:
        frequency = exp.ctx.targets[target].frequency
    if amplitude is None:
        amplitude = exp.ctx.params.control_amplitude[target]
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    detuning_range = np.asarray(detuning_range, dtype=float)
    time_range = np.asarray(time_range, dtype=float)

    raw_data_buffer: list[np.ndarray] = []

    for detuning in tqdm(detuning_range):
        with exp.ctx.util.no_output():
            sweep_result = exp.measurement_service.sweep_parameter(
                sequence=lambda t: {target: Rect(duration=t, amplitude=amplitude)},
                sweep_range=time_range,
                frequencies={target: frequency + detuning},
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )

        data = sweep_result.data[target]
        z = np.asarray(data.data, dtype=np.complex128)
        raw_data_buffer.append(z)

    # shape: (detuning, time)
    raw_data = np.array(raw_data_buffer)

    # Global PCA axis
    z_all = raw_data.ravel()
    z_centered = z_all - np.mean(z_all)
    iq_all = np.column_stack(
        [
            z_centered.real,
            z_centered.imag,
        ]
    )
    try:
        _, _, vh = np.linalg.svd(iq_all, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0])

    # Project all data using SAME axis
    mean_real = np.mean(z_all.real)
    mean_imag = np.mean(z_all.imag)
    projected_data = axis[0] * (raw_data.real - mean_real) + axis[1] * (
        raw_data.imag - mean_imag
    )
    if np.mean(projected_data[:, 0]) < np.mean(projected_data):
        projected_data *= -1

    # shape: (time, detuning)
    chevron_data = projected_data.T

    # --- plot ---
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=frequency + detuning_range,
            y=time_range,
            z=chevron_data,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Chevron pattern : {target}<br><sup>amplitude={amplitude:.6g}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Time (ns)",
        width=600,
        height=400,
    )

    if plot:
        fig.show()

    if save_image:
        viz.save_figure(
            fig,
            name=f"chevron_pattern_{target}",
            width=600,
            height=400,
        )

    return Result(
        data={
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequency": frequency,
            "amplitude": amplitude,
            "chevron_data": chevron_data,
        },
        figure=fig,
    )


def analyze_chevron_matched_transform(
    result: Result,
    target: str,
    *,
    omega_q_range: ArrayLike,
    omega_rabi_range: ArrayLike,
    subtract_mean: bool = True,
    refine_peak_quadratic: bool = False,
    quadratic_window: int = 2,
    plot: bool = True,
    save_image: bool = True,
) -> Result:
    """
    Apply a chevron matched transform and estimate the peak position.

    This function applies

        F(omega_q', omega_rabi')
        = integral d omega_d d tau
          f(omega_d, tau)
          cos(2 pi sqrt(omega_rabi'^2 + (omega_d - omega_q')^2) tau)

    to measured chevron data, estimates the peak position, and optionally
    displays the transform as a heatmap.

    Parameters
    ----------
    result:
        Chevron measurement result returned by either
        measure_chevron_pattern() or chevron_pattern().
    target:
        Target qubit label.
    omega_q_range:
        Trial qubit frequencies in GHz.
    omega_rabi_range:
        Trial resonant Rabi frequencies in GHz.
        Note: 1 GHz = 1 cycle/ns.
    subtract_mean:
        If True, subtract the mean over tau for each drive frequency.
        This usually makes the oscillatory part clearer.
    refine_peak_quadratic:
        If True, refine the peak position by fitting a local quadratic
        surface around the grid maximum.
    quadratic_window:
        Half-width of the local quadratic fitting window.
        quadratic_window=2 uses a 5x5 patch.
    plot:
        If True, show a heatmap with the estimated peak position.
    save_image :
        If True, save the generated figure.

    Returns
    -------
    Result
        data:
            {
                "omega_q": estimated qubit frequency in GHz,
                "omega_rabi": estimated resonant Rabi frequency in GHz,
                "peak": peak estimation result returned by
                    _estimate_chevron_transform_peak(),
                "peak_prominence_ratio":
                    confidence metric defined as the ratio between
                    the main transform peak and the largest competing peak,
                "omega_q_range": trial qubit frequency axis in GHz,
                "omega_rabi_range": trial Rabi frequency axis in GHz,
                "transform": matched-transform map,
            }
        figure:
            Matched-transform heatmap figure.
    """
    data = result.data

    time_range = np.asarray(data["time_range"], dtype=float)  # ns
    detuning_range = np.asarray(data["detuning_range"], dtype=float)  # GHz

    if "frequency" in data:
        base_frequency = float(data["frequency"])  # GHz
    elif "frequencies" in data:
        base_frequency = float(data["frequencies"][target])  # GHz
    else:
        raise ValueError(
            "Chevron result does not contain 'frequency' or 'frequencies'."
        )

    omega_d = base_frequency + detuning_range  # GHz

    chevron_data = data["chevron_data"]
    if isinstance(chevron_data, dict):
        f = np.asarray(
            chevron_data[target],
            dtype=float,
        )
    else:
        f = np.asarray(
            chevron_data,
            dtype=float,
        )

    # Expected shape: (len(time_range), len(detuning_range))
    if f.shape != (time_range.size, omega_d.size):
        raise ValueError(
            "Unexpected chevron_data shape. "
            f"Expected {(time_range.size, omega_d.size)}, got {f.shape}."
        )

    if subtract_mean:
        # Remove the DC component along tau for each drive frequency.
        f = f - np.nanmean(f, axis=0, keepdims=True)

    omega_q_range = np.asarray(omega_q_range, dtype=float)
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)

    transform = np.empty(
        (omega_rabi_range.size, omega_q_range.size),
        dtype=float,
    )

    tau = time_range[:, None, None]
    wd = omega_d[None, :, None]
    omega_q_mesh = omega_q_range[None, None, :]
    f_expanded = f[:, :, None]

    detuning2 = (wd - omega_q_mesh) ** 2

    for i, omega_rabi in enumerate(omega_rabi_range):
        omega_eff = np.sqrt(omega_rabi**2 + detuning2)
        phase = 2.0 * np.pi * omega_eff * tau
        kernel = np.cos(phase)

        integrand = f_expanded * kernel

        tmp = np.trapz(
            integrand,
            x=time_range,
            axis=0,
        )

        transform[i] = np.trapz(
            tmp,
            x=omega_d,
            axis=0,
        )

    if np.any(np.isfinite(transform)):
        max_abs_idx = np.nanargmax(np.abs(transform))
        max_abs_value = transform.ravel()[max_abs_idx]
        if max_abs_value < 0:
            transform *= -1

    peak = _estimate_chevron_transform_peak(
        transform,
        omega_q_range,
        omega_rabi_range,
        refine_quadratic=refine_peak_quadratic,
        window=quadratic_window,
        show_diagnostics=plot,
    )

    omega_q_hat = peak["omega_q"]
    omega_rabi_hat = peak["omega_rabi"]

    peak_idx = np.unravel_index(
        np.nanargmax(transform),
        transform.shape,
    )
    ii, jj = peak_idx
    peak_value = transform[ii, jj]
    mask = np.ones(transform.shape, dtype=bool)
    radius = 16
    mask[
        max(0, ii - radius) : min(transform.shape[0], ii + radius + 1),
        max(0, jj - radius) : min(transform.shape[1], jj + radius + 1),
    ] = False
    background = transform[mask]
    finite_background = background[np.isfinite(background)]
    if finite_background.size > 0:
        second_peak_value = np.nanmax(np.abs(finite_background))
        peak_prominence_ratio = peak_value / max(second_peak_value, 1e-12)
    else:
        second_peak_value = np.nan
        peak_prominence_ratio = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=omega_q_range,
            y=omega_rabi_range,
            z=transform,
            colorscale="Viridis",
            name="Matched transform",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[omega_q_hat],
            y=[omega_rabi_hat],
            mode="markers",
            marker=dict(
                symbol="x",
                size=10,
                color="red",
                line=dict(width=2),
            ),
            name=(
                f"Estimated peak<br>"
                f"ω_q={omega_q_hat:.9g} GHz<br>"
                f"ω_Rabi={omega_rabi_hat:.9g} GHz"
            ),
        )
    )

    x_min = np.min(omega_q_range)
    x_max = np.max(omega_q_range)
    x_center = 0.5 * (x_min + x_max)
    x_span = x_max - x_min
    central_min = x_center - 0.2 * x_span
    central_max = x_center + 0.2 * x_span
    if central_min <= omega_q_hat <= central_max:
        half_span = 0.25 * x_span
        fig.update_xaxes(
            range=[
                x_center - half_span,
                x_center + half_span,
            ]
        )

    y_min = np.min(omega_rabi_range)
    y_max = max(
        0.5 * np.max(omega_rabi_range),
        min(2.0 * omega_rabi_hat, np.max(omega_rabi_range)),
    )
    if y_max > y_min:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=dict(
            text=(
                f"Chevron matched transform : {target}"
                f"<br><sup>"
                f"ω_q={omega_q_hat:.9g} GHz, "
                f"ω_Rabi={omega_rabi_hat:.9g} GHz, "
                f"refined={peak['refined']}"
                f"</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=r"Trial qubit frequency ω'_q (GHz)",
        yaxis_title=r"Trial Rabi frequency ω'_Rabi (GHz)",
        width=650,
        height=500,
    )

    if plot:
        fig.show()

    if save_image:
        viz.save_figure(
            fig,
            name=f"chevron_matched_transform_{target}",
            width=650,
            height=500,
        )

    return Result(
        data={
            "omega_q": omega_q_hat,
            "omega_rabi": omega_rabi_hat,
            "peak": peak,
            "peak_prominence_ratio": peak_prominence_ratio,
            "omega_q_range": omega_q_range,
            "omega_rabi_range": omega_rabi_range,
            "transform": transform,
        },
        figure=fig,
    )


def _estimate_chevron_transform_peak(
    transform: ArrayLike,
    omega_q_range: ArrayLike,
    omega_rabi_range: ArrayLike,
    *,
    refine_quadratic: bool = False,
    window: int = 2,
    show_diagnostics: bool = False,
) -> dict:
    """
    Estimate (omega_q, omega_rabi) from a chevron matched-transform map.

    Parameters
    ----------
    transform:
        2D transform array with shape (len(omega_rabi_range), len(omega_q_range)).
    omega_q_range:
        Trial qubit frequency axis in GHz.
    omega_rabi_range:
        Trial Rabi frequency axis in GHz.
    refine_quadratic:
        If True, refine the maximum by fitting a local quadratic surface.
    window:
        Half-width of the fitting window. window=2 uses a 5x5 patch.
    show_diagnostics:
        If True, show diagnostics.

    Returns
    -------
    dict
        Estimated parameters and diagnostic information.
    """

    def _log(msg: str):
        if show_diagnostics:
            print(f"[peak_estimation] {msg}")

    z = np.asarray(transform, dtype=float)
    omega_q_range = np.asarray(omega_q_range, dtype=float)
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)

    if z.shape != (omega_rabi_range.size, omega_q_range.size):
        raise ValueError(
            "transform shape must be (len(omega_rabi_range), len(omega_q_range))."
        )

    if not np.any(np.isfinite(z)):
        raise ValueError("transform contains no finite values.")

    max_flat_index = np.nanargmax(z)
    i_max, j_max = np.unravel_index(max_flat_index, z.shape)

    omega_q_grid = omega_q_range[j_max]
    omega_rabi_grid = omega_rabi_range[i_max]
    z_max = z[i_max, j_max]

    result = {
        "omega_q": omega_q_grid,
        "omega_rabi": omega_rabi_grid,
        "omega_q_grid": omega_q_grid,
        "omega_rabi_grid": omega_rabi_grid,
        "z_max": z_max,
        "i_max": i_max,
        "j_max": j_max,
        "refined": False,
    }

    if not refine_quadratic:
        return result

    i0 = max(i_max - window, 0)
    i1 = min(i_max + window + 1, z.shape[0])
    j0 = max(j_max - window, 0)
    j1 = min(j_max + window + 1, z.shape[1])

    z_patch = z[i0:i1, j0:j1]
    r_patch = omega_rabi_range[i0:i1]
    q_patch = omega_q_range[j0:j1]

    if z_patch.shape[0] < 3 or z_patch.shape[1] < 3:
        _log("patch too small for quadratic fit")
        return result

    qq, rr = np.meshgrid(q_patch, r_patch)

    x = qq.ravel()
    y = rr.ravel()
    zz = z_patch.ravel()

    mask = np.isfinite(zz)
    x = x[mask]
    y = y[mask]
    zz = zz[mask]

    if zz.size < 6:
        _log("not enough valid points for quadratic fit")
        return result

    # Fit:
    # z = a x^2 + b y^2 + c x y + d x + e y + f
    # To improve numerical conditioning, use local coordinates.
    x0 = omega_q_grid
    y0 = omega_rabi_grid
    X = x - x0
    Y = y - y0

    A = np.column_stack(
        [
            X**2,
            Y**2,
            X * Y,
            X,
            Y,
            np.ones_like(X),
        ]
    )

    try:
        coeffs, *_ = np.linalg.lstsq(A, zz, rcond=None)
    except np.linalg.LinAlgError:
        _log("least squares failed")
        return result

    a, b, c, d, e, f0 = coeffs

    # Stationary point:
    # 2aX + cY + d = 0
    # cX + 2bY + e = 0
    H = np.array([[2 * a, c], [c, 2 * b]], dtype=float)
    g = np.array([d, e], dtype=float)

    try:
        X_peak, Y_peak = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        _log("failed to solve for stationary point (singular Hessian)")
        return result

    omega_q_refined = x0 + X_peak
    omega_rabi_refined = y0 + Y_peak

    # Accept refinement only if the peak remains inside the fitted patch.
    if not (q_patch.min() <= omega_q_refined <= q_patch.max()):
        _log("refined omega_q outside patch → rejected")
        return result
    if not (r_patch.min() <= omega_rabi_refined <= r_patch.max()):
        _log("refined omega_rabi outside patch → rejected")
        return result

    # Optional: check that the quadratic surface is locally concave.
    eigvals = np.linalg.eigvalsh(H)
    is_local_maximum = np.all(eigvals < 0)

    z_refined = (
        a * X_peak**2
        + b * Y_peak**2
        + c * X_peak * Y_peak
        + d * X_peak
        + e * Y_peak
        + f0
    )

    if not is_local_maximum:
        _log(f"not a local maximum (eigvals={eigvals})")
        return result

    result.update(
        {
            "omega_q": omega_q_refined,
            "omega_rabi": omega_rabi_refined,
            "z_refined": z_refined,
            "quadratic_coeffs": coeffs,
            "refined": True,
            "fit_window": window,
        }
    )

    return result
