"""Chevron-based qubit frequency calibration using matched-transform analysis."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from qxpulse import Rect, get_sampling_period

import qubex.visualization as viz
from qubex.compat.numpy_compat import trapezoid
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
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
    background_radius: int | None = None,
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
    this function does not perform coarse-search measurements or adaptive
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
        Default: np.linspace(-0.05, 0.05, 41)
    time_range : array-like, optional
        Pulse duration range (ns).
        Default: np.linspace(0, 250, 26)
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
        Default: max(1, DEFAULT_SHOTS // 4)
    shot_interval : float, optional
        Interval between shots.
    quadratic_window : int, optional
        Half-width of the local quadratic fitting window used for
        sub-grid peak refinement.
        Default: 3.
        quadratic_window=3 uses a 7x7 patch.
    background_radius : int, optional
        Radius, in transform-grid points, around the detected peak to exclude
        when estimating background RMS.
        Default: 32.
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
                        "peak_background_rms_ratio":
                            confidence metric defined as the ratio between
                            the main transform peak and background RMS outside
                            a neighborhood of the main peak,
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
                "peak_background_rms_ratios":
                    peak-to-background-RMS ratios for each target,
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
    - Quadratic refinement improves sub-grid peak estimation accuracy.
    - The peak-to-background-RMS ratio serves as a confidence metric for
        peak detection.
    """
    if targets is None:
        targets = exp.ctx.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 41)
    if time_range is None:
        time_range = np.linspace(0, 250, 26)
    if omega_rabi_range is None:
        omega_rabi_range = 0.1 * (10 ** np.linspace(0, 1, 256) - 1) / 9
    if n_shots is None:
        n_shots = max(1, DEFAULT_SHOTS // 4)
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if quadratic_window is None:
        quadratic_window = 3
    if background_radius is None:
        background_radius = 32
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
    peak_background_rms_ratios: dict[str, float] = {}
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
            background_radius=background_radius,
            plot=plot,
            save_image=save_image,
        )

        analysis = analysis_result.data

        omega_q = analysis["omega_q"]
        omega_rabi = analysis["omega_rabi"]
        peak_background_rms_ratio = analysis["peak_background_rms_ratio"]

        print(
            f"[RESULT] ω_q={omega_q:.6f} GHz,"
            f" ω_Rabi={omega_rabi:.6f} GHz"
            f" peak_background_rms_ratio={peak_background_rms_ratio:.3f}"
        )

        results_data[target] = {
            "omega_q": omega_q,
            "omega_rabi": omega_rabi,
            "peak_background_rms_ratio": peak_background_rms_ratio,
            "frequency_used": frequencies[target],
            "amplitude_used": amplitudes[target],
            "chevron_data": measurement_result.data["chevron_data"],
            "transform": analysis["transform"],
        }

        resonant_frequencies[target] = omega_q
        omega_rabis[target] = omega_rabi
        peak_background_rms_ratios[target] = peak_background_rms_ratio
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
            "peak_background_rms_ratios": peak_background_rms_ratios,
        },
        figures=figures,
    )


def estimate_qubit_frequency_from_chevron_adaptive(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    search_detuning_width: float | None = None,
    search_duration_max: float | None = None,
    search_detuning_points_per_half: int | None = None,
    search_duration_points: int | None = None,
    search_duration_alpha: float | None = None,
    peak_background_rms_threshold: float | None = None,
    final_detuning_range: ArrayLike | None = None,
    final_time_range: ArrayLike | None = None,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    omega_rabi_range: ArrayLike | None = None,
    target_omega_rabi: float | None = None,
    target_amplitude_scale_bounds: tuple[float, float] | None = None,
    n_shots: int | None = None,
    search_n_shots: int | None = None,
    shot_interval: float | None = None,
    final_quadratic_window: int | None = None,
    background_radius: int | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Adaptively estimate qubit frequencies with a coarse chevron first pass.

    The first pass measures a coarse chevron over a wide detuning-duration
    domain with nonlinear duration spacing. If the peak/background-RMS ratio
    is small, additional chevron data are measured on both detuning-side
    bands, then the combined raw IQ data are projected once before
    re-analysis. The final measurement uses the regular chevron grid with
    amplitude rescaled from the coarse-search Rabi estimate.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context.
    targets : Collection[str] or str, optional
        Target qubit label(s).
        If None, all available qubits are used.
    search_detuning_width : float, optional
        Width of the initial central search detuning interval in GHz.
        The first pass samples
        [-search_detuning_width / 2, search_detuning_width / 2].
        Default: 0.2.
    search_duration_max : float, optional
        Nominal maximum pulse duration in ns for search measurements.
        The generated axis is rounded to the qxpulse sampling period.
        Default: 400.0.
    search_detuning_points_per_half : int, optional
        Number of detuning intervals per half-width in the initial search.
        The initial detuning axis has 2 * search_detuning_points_per_half + 1
        points. If side-band extension is triggered, the doubled range uses
        twice this value per half-width.
        Default: 20.
    search_duration_points : int, optional
        Number of duration intervals in the initial search.
        The generated duration axis has search_duration_points + 1 samples
        before sampling-period rounding.
        Default: 15.
    search_duration_alpha : float, optional
        Exponent for nonlinear search duration spacing:
        tau_k = search_duration_max * (k / N) ** search_duration_alpha.
        Default: 1.5.
    peak_background_rms_threshold : float, optional
        Minimum search peak/background-RMS ratio required to skip
        side-band extension.
        Default: 8.0.
    final_detuning_range : array-like, optional
        Detuning range (GHz) for the final regular chevron measurement.
        Default: np.linspace(-0.05, 0.05, 41).
    final_time_range : array-like, optional
        Pulse duration range (ns) for the final regular chevron measurement.
        Default: np.linspace(0, 250, 26).
    frequencies : dict[str, float], optional
        Initial drive frequencies (GHz) for each target qubit.
        If None, calibrated qubit frequencies are used.
    amplitudes : dict[str, float], optional
        Initial drive amplitudes for each target qubit.
        If None, default control amplitudes are used.
    omega_rabi_range : array-like, optional
        Trial resonant Rabi frequencies (GHz) used in matched transforms.
    target_omega_rabi : float, optional
        Target resonant Rabi frequency (GHz) used to rescale the final
        measurement amplitude from the coarse-search estimate.
        Default: 0.0125.
    target_amplitude_scale_bounds : tuple[float, float], optional
        Lower and upper clipping bounds for amplitude rescaling.
        Used both for the search-to-final amplitude update and for the
        clipped target_amplitudes recommendation.
        Default: (0.1, 10.0).
    n_shots : int, optional
        Number of shots for the final regular chevron measurement.
        Default: max(1, DEFAULT_SHOTS // 4).
    search_n_shots : int, optional
        Number of shots for search measurements.
        Default: n_shots.
    shot_interval : float, optional
        Interval between shots.
    final_quadratic_window : int, optional
        Half-width of the local quadratic fitting window used for
        final sub-grid peak refinement. The search stage uses 2.
        Default: 3.
    background_radius : int, optional
        Radius, in transform-grid points, around the detected peak to exclude
        when estimating background RMS.
        Default: 32.
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
                        "omega_q": final estimated qubit frequency (GHz),
                        "omega_rabi": final estimated Rabi frequency (GHz),
                        "peak_background_rms_ratio":
                            final peak-to-background-RMS ratio,
                        "frequency_used": final drive frequency used (GHz),
                        "amplitude_used": final drive amplitude used,
                        "chevron_data": final projected chevron data,
                        "transform": final matched-transform map,
                    }
                },
                "time_range": final pulse duration axis (ns),
                "detuning_range": final detuning axis (GHz),
                "resonant_frequencies": final estimated qubit frequencies,
                "target_amplitudes":
                    clipped amplitudes estimated to realize target_omega_rabi
                    from the final measurement,
                "peak_background_rms_ratios":
                    final peak-to-background-RMS ratios for each target,
                "search_results":
                    search estimates, measured data, amplitude scale,
                    and whether side-band extension was used,
            }
        figures:
            {
                f"{target}_search_measurement": search chevron figure,
                f"{target}_search_transform": search transform figure,
                f"{target}_measurement": final chevron measurement figure,
                f"{target}_transform": final transform figure,
            }

    Notes
    -----
    - If the initial search ratio is below threshold, additional samples are
        measured on the outer detuning bands of the doubled detuning range.
    - The final drive frequency is set to the search omega_q estimate.
    - The final drive amplitude is multiplied by
        target_omega_rabi / omega_rabi_est, clipped to
        target_amplitude_scale_bounds.
    """
    if targets is None:
        targets = exp.ctx.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if search_detuning_width is None:
        search_detuning_width = 0.2
    if search_duration_max is None:
        search_duration_max = 400.0
    if search_detuning_points_per_half is None:
        search_detuning_points_per_half = 20
    if search_duration_points is None:
        search_duration_points = 15
    if search_duration_alpha is None:
        search_duration_alpha = 1.5
    if peak_background_rms_threshold is None:
        peak_background_rms_threshold = 8.0
    if final_detuning_range is None:
        final_detuning_range = np.linspace(-0.05, 0.05, 41)
    if final_time_range is None:
        final_time_range = np.linspace(0, 250, 26)
    if omega_rabi_range is None:
        omega_rabi_range = 0.1 * (10 ** np.linspace(0, 1, 256) - 1) / 9
    if target_omega_rabi is None:
        target_omega_rabi = 0.0125
    if target_amplitude_scale_bounds is None:
        target_amplitude_scale_bounds = (0.1, 10.0)
    if n_shots is None:
        n_shots = max(1, DEFAULT_SHOTS // 4)
    if search_n_shots is None:
        search_n_shots = n_shots
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if final_quadratic_window is None:
        final_quadratic_window = 3
    if background_radius is None:
        background_radius = 32
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    search_detuning_width = float(search_detuning_width)
    search_duration_max = float(search_duration_max)
    search_detuning_points_per_half = int(search_detuning_points_per_half)
    search_duration_points = int(search_duration_points)
    search_duration_alpha = float(search_duration_alpha)
    peak_background_rms_threshold = float(peak_background_rms_threshold)
    if len(target_amplitude_scale_bounds) != 2:
        raise ValueError("target_amplitude_scale_bounds must contain two values.")
    target_amplitude_scale_bounds = (
        float(target_amplitude_scale_bounds[0]),
        float(target_amplitude_scale_bounds[1]),
    )
    final_detuning_range = np.asarray(final_detuning_range, dtype=float)
    final_time_range = np.asarray(final_time_range, dtype=float)
    if final_time_range.size < 2:
        raise ValueError("final_time_range must contain at least two points.")
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)
    if search_detuning_width <= 0:
        raise ValueError("search_detuning_width must be positive.")
    if search_duration_max <= 0:
        raise ValueError("search_duration_max must be positive.")
    if search_detuning_points_per_half <= 0:
        raise ValueError("search_detuning_points_per_half must be positive.")
    if search_duration_points <= 0:
        raise ValueError("search_duration_points must be positive.")
    if search_duration_alpha <= 0:
        raise ValueError("search_duration_alpha must be positive.")
    scale_min, scale_max = target_amplitude_scale_bounds
    if scale_min <= 0 or scale_max <= 0:
        raise ValueError("target_amplitude_scale_bounds must be positive.")
    if scale_min > scale_max:
        raise ValueError(
            "target_amplitude_scale_bounds lower bound must be less than or "
            "equal to upper bound."
        )

    search_detuning_range, search_time_range = _make_adaptive_search_ranges(
        detuning_width=search_detuning_width,
        detuning_points_per_half=search_detuning_points_per_half,
        duration_max=search_duration_max,
        duration_points=search_duration_points,
        duration_alpha=search_duration_alpha,
    )

    if frequencies is None:
        frequencies = {target: exp.ctx.targets[target].frequency for target in targets}

    if amplitudes is None:
        amplitudes = {
            target: exp.ctx.params.control_amplitude[target] for target in targets
        }

    results_data = {}
    search_results = {}
    resonant_frequencies: dict[str, float] = {}
    target_amplitudes: dict[str, float] = {}
    peak_background_rms_ratios: dict[str, float] = {}
    figures = {}

    for target in targets:
        freq = frequencies[target]
        amp = amplitudes[target]

        print(f"=== Target: {target} ===")
        print("[search measurement]")

        search_measurement, search_analysis = _measure_and_analyze_chevron(
            exp,
            target,
            detuning_range=search_detuning_range,
            time_range=search_time_range,
            frequency=freq,
            amplitude=amp,
            omega_rabi_range=omega_rabi_range,
            n_shots=search_n_shots,
            shot_interval=shot_interval,
            refine_peak_quadratic=True,
            quadratic_window=2,
            background_radius=background_radius,
            plot=plot,
            save_image=False,
        )
        search_analysis_data = search_analysis.data
        extended_search = False

        if (
            search_analysis_data["peak_background_rms_ratio"]
            < peak_background_rms_threshold
        ):
            print("[search extension]")
            extended_detuning_range = np.linspace(
                -search_detuning_width,
                search_detuning_width,
                4 * search_detuning_points_per_half + 1,
            )
            lower_side, upper_side = _split_unmeasured_detuning_side_bands(
                measured_detuning_range=search_detuning_range,
                extended_detuning_range=extended_detuning_range,
            )
            side_results = [
                measure_chevron_pattern(
                    exp,
                    target,
                    detuning_range=detuning_side_range,
                    time_range=search_time_range,
                    frequency=freq,
                    amplitude=amp,
                    n_shots=search_n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                    save_image=False,
                )
                for detuning_side_range in (lower_side, upper_side)
            ]
            search_measurement = _combine_chevron_measurements(
                [search_measurement, *side_results],
                target=target,
                frequency=freq,
                amplitude=amp,
                plot=plot,
                save_image=False,
            )
            omega_q_range = np.linspace(
                freq + 2 * np.min(extended_detuning_range),
                freq + 2 * np.max(extended_detuning_range),
                1024,
            )
            search_analysis = analyze_chevron_matched_transform(
                search_measurement,
                target,
                omega_q_range=omega_q_range,
                omega_rabi_range=omega_rabi_range,
                refine_peak_quadratic=True,
                quadratic_window=2,
                background_radius=background_radius,
                plot=plot,
                save_image=False,
            )
            search_analysis_data = search_analysis.data
            extended_search = True

        omega_q_est = search_analysis_data["omega_q"]
        omega_rabi_est = search_analysis_data["omega_rabi"]
        peak_background_rms_ratio_search = search_analysis_data[
            "peak_background_rms_ratio"
        ]

        print(
            f"[search result] ω_q={omega_q_est:.6f},"
            f" ω_Rabi={omega_rabi_est:.6f},"
            f" peak_background_rms_ratio={peak_background_rms_ratio_search:.3f}"
        )

        freq_old = freq
        amp_old = amp
        freq = omega_q_est

        scale = target_omega_rabi / max(omega_rabi_est, 1e-12)
        scale = np.clip(scale, scale_min, scale_max)
        amp *= scale

        print(
            f"[update] freq: {freq_old:.6f} -> {freq:.6f}, "
            f"amp: {amp_old:.6g} -> {amp:.6g} (scale: {scale:.3f})"
        )

        search_results[target] = {
            "omega_q": omega_q_est,
            "omega_rabi": omega_rabi_est,
            "peak_background_rms_ratio": peak_background_rms_ratio_search,
            "frequency_used": freq_old,
            "amplitude_used": amp_old,
            "scale": scale,
            "extended": extended_search,
            "detuning_range": search_measurement.data["detuning_range"],
            "time_range": search_measurement.data["time_range"],
            "chevron_data": search_measurement.data["chevron_data"],
            "transform": search_analysis_data["transform"],
        }

        figures[f"{target}_search_measurement"] = search_measurement.figure
        figures[f"{target}_search_transform"] = search_analysis.figure

        print("[final measurement]")

        measurement_result, analysis_result = _measure_and_analyze_chevron(
            exp,
            target,
            detuning_range=final_detuning_range,
            time_range=final_time_range,
            frequency=freq,
            amplitude=amp,
            omega_rabi_range=omega_rabi_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            refine_peak_quadratic=True,
            quadratic_window=final_quadratic_window,
            background_radius=background_radius,
            plot=plot,
            save_image=save_image,
        )
        analysis = analysis_result.data

        omega_q_final = analysis["omega_q"]
        omega_rabi_final = analysis["omega_rabi"]
        peak_background_rms_ratio_final = analysis["peak_background_rms_ratio"]

        print(
            f"[FINAL] ω_q={omega_q_final:.6f} GHz,"
            f" ω_Rabi={omega_rabi_final:.6f} GHz,"
            f" peak_background_rms_ratio={peak_background_rms_ratio_final:.3f}"
        )

        target_amplitude_scale = target_omega_rabi / max(omega_rabi_final, 1e-12)
        target_amplitude_scale = np.clip(
            target_amplitude_scale,
            scale_min,
            scale_max,
        )
        target_amplitude = amp * target_amplitude_scale
        print(
            f"[target amplitude] amp={target_amplitude:.6g}"
            f" for target_omega_rabi={target_omega_rabi:.6f} GHz"
            f" (scale: {target_amplitude_scale:.3f})"
        )

        results_data[target] = {
            "omega_q": omega_q_final,
            "omega_rabi": omega_rabi_final,
            "peak_background_rms_ratio": peak_background_rms_ratio_final,
            "frequency_used": freq,
            "amplitude_used": amp,
            "chevron_data": measurement_result.data["chevron_data"],
            "transform": analysis["transform"],
        }

        resonant_frequencies[target] = omega_q_final
        target_amplitudes[target] = target_amplitude
        peak_background_rms_ratios[target] = peak_background_rms_ratio_final
        figures[f"{target}_measurement"] = measurement_result.figure
        figures[f"{target}_transform"] = analysis_result.figure

    return Result(
        data={
            "results": results_data,
            "time_range": final_time_range,
            "detuning_range": final_detuning_range,
            "resonant_frequencies": resonant_frequencies,
            "target_amplitudes": target_amplitudes,
            "peak_background_rms_ratios": peak_background_rms_ratios,
            "search_results": search_results,
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
    background_radius: int,
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

    # The matched-transform resonance search intentionally spans about twice
    # the measured detuning interval so a peak near the edge is still captured.
    omega_q_range = np.linspace(
        frequency + 2 * np.min(np.asarray(detuning_range)),
        frequency + 2 * np.max(np.asarray(detuning_range)),
        1024,
    )

    analysis_result = analyze_chevron_matched_transform(
        measurement_result,
        target,
        omega_q_range=omega_q_range,
        omega_rabi_range=omega_rabi_range,
        refine_peak_quadratic=refine_peak_quadratic,
        quadratic_window=quadratic_window,
        background_radius=background_radius,
        plot=plot,
        save_image=save_image,
    )

    return measurement_result, analysis_result


def _make_adaptive_search_ranges(
    *,
    detuning_width: float,
    detuning_points_per_half: int,
    duration_max: float,
    duration_points: int,
    duration_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    detuning_range = np.linspace(
        -0.5 * detuning_width,
        0.5 * detuning_width,
        2 * detuning_points_per_half + 1,
    )
    k = np.arange(duration_points + 1, dtype=float)
    time_range = duration_max * (k / duration_points) ** duration_alpha
    sampling_period = get_sampling_period()
    time_range = np.round(time_range / sampling_period) * sampling_period
    if time_range.size < 2:
        raise ValueError("adaptive search time_range must contain at least two points.")
    return detuning_range, time_range


def _split_unmeasured_detuning_side_bands(
    *,
    measured_detuning_range: ArrayLike,
    extended_detuning_range: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    measured_detuning_range = np.asarray(measured_detuning_range, dtype=float)
    extended_detuning_range = np.asarray(extended_detuning_range, dtype=float)
    measured_min = np.min(measured_detuning_range)
    measured_max = np.max(measured_detuning_range)
    lower_side = extended_detuning_range[extended_detuning_range < measured_min]
    upper_side = extended_detuning_range[extended_detuning_range > measured_max]
    if lower_side.size == 0 or upper_side.size == 0:
        raise ValueError("extended_detuning_range must include unmeasured side bands.")
    return lower_side, upper_side


def _combine_chevron_measurements(
    results: Collection[Result],
    *,
    target: str,
    frequency: float,
    amplitude: float,
    plot: bool,
    save_image: bool,
) -> Result:
    """Combine chevron payloads and project all raw IQ data at once."""
    results = list(results)
    if not results:
        raise ValueError("results must contain at least one measurement result.")

    time_range = np.asarray(results[0].data["time_range"], dtype=float)
    detuning_parts = []
    raw_parts = []
    for result in results:
        result_time_range = np.asarray(result.data["time_range"], dtype=float)
        if result_time_range.shape != time_range.shape or not np.allclose(
            result_time_range,
            time_range,
        ):
            raise ValueError("all chevron results must use the same time_range.")
        detuning_parts.append(np.asarray(result.data["detuning_range"], dtype=float))
        try:
            raw_parts.append(np.asarray(result.data["raw_data"], dtype=np.complex128))
        except KeyError as exc:
            raise ValueError(
                "chevron measurement results must contain raw_data for "
                "consistent combined PCA projection."
            ) from exc

    detuning_range = np.concatenate(detuning_parts)
    raw_data = np.concatenate(raw_parts, axis=0)
    if raw_data.shape != (detuning_range.size, time_range.size):
        raise ValueError(
            "combined raw_data must have shape (len(detuning_range), len(time_range))."
        )

    sort_index = np.argsort(detuning_range)
    detuning_range = detuning_range[sort_index]
    raw_data = raw_data[sort_index]
    if np.any(np.diff(detuning_range) <= 0):
        raise ValueError("combined detuning_range must not contain duplicates.")

    chevron_data = _project_chevron_raw_data(raw_data)
    fig = _make_chevron_figure(
        target=target,
        detuning_range=detuning_range,
        time_range=time_range,
        chevron_data=chevron_data,
        frequency=frequency,
        amplitude=amplitude,
        title="Extended search chevron pattern",
    )

    if plot:
        fig.show()

    if save_image:
        viz.save_figure(
            fig,
            name=f"extended_search_chevron_pattern_{target}",
            width=600,
            height=400,
        )

    return Result(
        data={
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequency": frequency,
            "amplitude": amplitude,
            "raw_data": raw_data,
            "chevron_data": chevron_data,
        },
        figure=fig,
    )


def _project_chevron_raw_data(raw_data: ArrayLike) -> np.ndarray:
    """Project gridded chevron IQ data onto one PCA axis."""
    raw_data = np.asarray(raw_data, dtype=np.complex128)
    if raw_data.ndim != 2:
        raise ValueError("raw_data must be a two-dimensional array.")

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

    mean_real = np.mean(z_all.real)
    mean_imag = np.mean(z_all.imag)
    projected_data = axis[0] * (raw_data.real - mean_real) + axis[1] * (
        raw_data.imag - mean_imag
    )
    if np.mean(projected_data[:, 0]) < np.mean(projected_data):
        projected_data *= -1

    return projected_data.T


def _make_chevron_figure(
    *,
    target: str,
    detuning_range: ArrayLike,
    time_range: ArrayLike,
    chevron_data: ArrayLike,
    frequency: float,
    amplitude: float,
    title: str,
) -> go.Figure:
    detuning_range = np.asarray(detuning_range, dtype=float)
    time_range = np.asarray(time_range, dtype=float)
    chevron_data = np.asarray(chevron_data, dtype=float)

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
            text=f"{title} : {target}<br><sup>amplitude={amplitude:.6g}</sup>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Time (ns)",
        width=600,
        height=400,
    )
    return fig


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
    target : str
        Target qubit label.
    detuning_range : array-like, optional
        Frequency detuning range (GHz).
        Default: np.linspace(-0.05, 0.05, 41).
    time_range : array-like, optional
        Pulse duration range (ns).
        Default: np.linspace(0, 250, 26).
    frequency : float, optional
        Base frequency (GHz).
        If None, the calibrated qubit frequency is used.
    amplitude : float, optional
        Drive amplitude.
        If None, the default control amplitude is used.
    n_shots : int, optional
        Number of shots.
        Default: max(1, DEFAULT_SHOTS // 4).
    shot_interval : float, optional
        Interval between shots.
    plot : bool, optional
        If True, display heatmap.
    save_image : bool, optional
        If True, save the generated heatmap figure.

    Returns
    -------
    Result
        data = {
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequency": frequency,
            "amplitude": amplitude,
            "raw_data":
                raw IQ data with shape (len(detuning_range), len(time_range)),
            "chevron_data":
                projected chevron data with shape
                (len(time_range), len(detuning_range)),
        }
        figure:
            Chevron heatmap figure.

    Notes
    -----
    - Each pulse is generated as Rect(duration, amplitude).detuned(detuning)
        while the carrier frequency is kept at frequency.
    - Data are projected onto a global PCA axis in the IQ plane.
    """
    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 41)
    if time_range is None:
        time_range = np.linspace(0, 250, 26)
    if frequency is None:
        frequency = exp.ctx.targets[target].frequency
    if amplitude is None:
        amplitude = exp.ctx.params.control_amplitude[target]
    if n_shots is None:
        n_shots = max(1, DEFAULT_SHOTS // 4)
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    detuning_range = np.asarray(detuning_range, dtype=float)
    time_range = np.asarray(time_range, dtype=float)

    sweep_points = [
        (detuning, duration) for detuning in detuning_range for duration in time_range
    ]

    def sequence(point_index: int) -> dict[str, Rect]:
        detuning, duration = sweep_points[int(point_index)]
        pulse = Rect(duration=duration, amplitude=amplitude).detuned(detuning)
        return {target: pulse}

    with exp.ctx.util.no_output():
        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(sweep_points)),
            frequencies={target: frequency},
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )

    data = sweep_result.data[target]
    z = np.asarray(data.data, dtype=np.complex128)

    # shape: (detuning, time)
    raw_data = z.reshape(detuning_range.size, time_range.size)
    chevron_data = _project_chevron_raw_data(raw_data)

    fig = _make_chevron_figure(
        target=target,
        detuning_range=detuning_range,
        time_range=time_range,
        chevron_data=chevron_data,
        frequency=frequency,
        amplitude=amplitude,
        title="Chevron pattern",
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
            "raw_data": raw_data,
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
    refine_peak_quadratic: bool = True,
    quadratic_window: int = 3,
    background_radius: int = 32,
    plot: bool = True,
    save_image: bool = True,
) -> Result:
    """
    Apply a chevron matched transform and estimate the peak position.

    This function applies a weighted-integral matched transform to measured
    chevron data, estimates the peak position, and optionally displays the
    transform as a heatmap:

        F(omega_q', omega_rabi')
        = integral d omega_d d tau
          f(omega_d, tau)
          cos(2 pi sqrt(omega_rabi'^2 + (omega_d - omega_q')^2) tau)

    Parameters
    ----------
    result:
        Chevron measurement result returned by either
        measure_chevron_pattern() or an API with compatible payload keys.
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
        Default: True.
    refine_peak_quadratic:
        If True, refine the peak position by fitting a local quadratic
        surface around the grid maximum.
        Default: True.
    quadratic_window:
        Half-width of the local quadratic fitting window.
        Default: 3.
        quadratic_window=3 uses a 7x7 patch.
    background_radius:
        Radius around the peak excluded from background RMS estimation.
        Default: 32.
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
                "peak_background_rms_ratio":
                    confidence metric defined as the ratio between
                    the main transform peak and the background RMS,
                "peak_background_rms": background RMS used in the ratio,
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

        tmp = trapezoid(
            integrand,
            x=time_range,
            axis=0,
        )

        transform[i] = trapezoid(
            tmp,
            x=omega_d,
            axis=0,
        )

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

    background_rms, peak_background_rms_ratio = _compute_peak_background_rms_ratio(
        transform=transform,
        background_radius=background_radius,
    )

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
                f"peak/background_rms={peak_background_rms_ratio:.3g}, "
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
            "peak_background_rms": background_rms,
            "peak_background_rms_ratio": peak_background_rms_ratio,
            "omega_q_range": omega_q_range,
            "omega_rabi_range": omega_rabi_range,
            "transform": transform,
        },
        figure=fig,
    )


def _compute_peak_background_rms_ratio(
    *,
    transform: ArrayLike,
    background_radius: int,
) -> tuple[float, float]:
    """Return background RMS and peak/background-RMS ratio for a transform map."""
    z = np.asarray(transform, dtype=float)
    if background_radius < 0:
        raise ValueError("background_radius must be non-negative.")
    if not np.any(np.isfinite(z)):
        return np.nan, np.nan

    peak_idx = np.unravel_index(np.nanargmax(z), z.shape)
    ii, jj = peak_idx
    peak_value = z[ii, jj]
    radius = int(background_radius)

    mask = np.ones(z.shape, dtype=bool)
    mask[
        max(0, ii - radius) : min(z.shape[0], ii + radius + 1),
        max(0, jj - radius) : min(z.shape[1], jj + radius + 1),
    ] = False

    finite_background = z[mask]
    finite_background = finite_background[np.isfinite(finite_background)]
    if finite_background.size == 0:
        return np.nan, np.nan

    background_rms = float(np.sqrt(np.nanmean(finite_background**2)))
    peak_background_rms_ratio = peak_value / max(background_rms, 1e-12)
    return background_rms, peak_background_rms_ratio


def _estimate_chevron_transform_peak(
    transform: ArrayLike,
    omega_q_range: ArrayLike,
    omega_rabi_range: ArrayLike,
    *,
    refine_quadratic: bool = False,
    window: int = 3,
    show_diagnostics: bool = True,
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
        Half-width of the fitting window. window=3 uses a 7x7 patch.
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
