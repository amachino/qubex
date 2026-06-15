"""EF/FH chevron measurement helpers."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from qxpulse import get_sampling_period

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, PulseSchedule

from .chevron_matched_transform import analyze_chevron_matched_transform


def estimate_fh_frequency_from_chevron(
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
    """Estimate FH resonance frequencies from FH chevron measurements."""
    targets = _normalize_fh_targets(exp, targets)

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
    frequencies = _normalize_fh_frequencies(exp, targets, frequencies)
    amplitudes = _normalize_fh_amplitudes(exp, targets, amplitudes)

    results_data = {}
    resonant_frequencies: dict[str, float] = {}
    omega_rabis: dict[str, float] = {}
    peak_background_rms_ratios: dict[str, float] = {}
    figures = {}

    for target in targets:
        print(f"=== Target: {target} ===")

        measurement_result, analysis_result = _measure_and_analyze_fh_chevron(
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
            f"[RESULT] omega_fh={omega_q:.6f} GHz,"
            f" omega_Rabi={omega_rabi:.6f} GHz,"
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


def estimate_fh_frequency_from_chevron_adaptive(
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
    """Adaptively estimate FH resonance frequencies with chevron measurements."""
    targets = _normalize_fh_targets(exp, targets)

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
    scale_min, scale_max = (
        float(target_amplitude_scale_bounds[0]),
        float(target_amplitude_scale_bounds[1]),
    )
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
    if scale_min <= 0 or scale_max <= 0:
        raise ValueError("target_amplitude_scale_bounds must be positive.")
    if scale_min > scale_max:
        raise ValueError(
            "target_amplitude_scale_bounds lower bound must be <= upper bound."
        )

    final_detuning_range = np.asarray(final_detuning_range, dtype=float)
    final_time_range = np.asarray(final_time_range, dtype=float)
    if final_time_range.size < 2:
        raise ValueError("final_time_range must contain at least two points.")
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)
    search_detuning_range, search_time_range = _make_adaptive_search_ranges(
        detuning_width=search_detuning_width,
        detuning_points_per_half=search_detuning_points_per_half,
        duration_max=search_duration_max,
        duration_points=search_duration_points,
        duration_alpha=search_duration_alpha,
    )
    frequencies = _normalize_fh_frequencies(exp, targets, frequencies)
    amplitudes = _normalize_fh_amplitudes(exp, targets, amplitudes)

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

        search_measurement, search_analysis = _measure_and_analyze_fh_chevron(
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
                measure_fh_chevron_pattern(
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
            f"[search result] omega_fh={omega_q_est:.6f},"
            f" omega_Rabi={omega_rabi_est:.6f},"
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
        measurement_result, analysis_result = _measure_and_analyze_fh_chevron(
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
            f"[FINAL] omega_fh={omega_q_final:.6f} GHz,"
            f" omega_Rabi={omega_rabi_final:.6f} GHz,"
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


def estimate_ef_frequency_from_chevron(
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
    """Estimate EF resonance frequencies from EF chevron measurements."""
    targets = _normalize_ef_targets(exp, targets)

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
    frequencies = _normalize_ef_frequencies(exp, targets, frequencies)
    amplitudes = _normalize_ef_amplitudes(exp, targets, amplitudes)

    results_data = {}
    resonant_frequencies: dict[str, float] = {}
    omega_rabis: dict[str, float] = {}
    peak_background_rms_ratios: dict[str, float] = {}
    figures = {}

    for target in targets:
        print(f"=== Target: {target} ===")

        measurement_result, analysis_result = _measure_and_analyze_ef_chevron(
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
            f"[RESULT] omega_ef={omega_q:.6f} GHz,"
            f" omega_Rabi={omega_rabi:.6f} GHz,"
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


def estimate_ef_frequency_from_chevron_adaptive(
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
    """Adaptively estimate EF resonance frequencies with chevron measurements."""
    targets = _normalize_ef_targets(exp, targets)

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
    scale_min, scale_max = (
        float(target_amplitude_scale_bounds[0]),
        float(target_amplitude_scale_bounds[1]),
    )
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
    if scale_min <= 0 or scale_max <= 0:
        raise ValueError("target_amplitude_scale_bounds must be positive.")
    if scale_min > scale_max:
        raise ValueError(
            "target_amplitude_scale_bounds lower bound must be <= upper bound."
        )

    final_detuning_range = np.asarray(final_detuning_range, dtype=float)
    final_time_range = np.asarray(final_time_range, dtype=float)
    if final_time_range.size < 2:
        raise ValueError("final_time_range must contain at least two points.")
    omega_rabi_range = np.asarray(omega_rabi_range, dtype=float)
    search_detuning_range, search_time_range = _make_adaptive_search_ranges(
        detuning_width=search_detuning_width,
        detuning_points_per_half=search_detuning_points_per_half,
        duration_max=search_duration_max,
        duration_points=search_duration_points,
        duration_alpha=search_duration_alpha,
    )
    frequencies = _normalize_ef_frequencies(exp, targets, frequencies)
    amplitudes = _normalize_ef_amplitudes(exp, targets, amplitudes)

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

        search_measurement, search_analysis = _measure_and_analyze_ef_chevron(
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
                measure_ef_chevron_pattern(
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
            f"[search result] omega_ef={omega_q_est:.6f},"
            f" omega_Rabi={omega_rabi_est:.6f},"
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
        measurement_result, analysis_result = _measure_and_analyze_ef_chevron(
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
            f"[FINAL] omega_ef={omega_q_final:.6f} GHz,"
            f" omega_Rabi={omega_rabi_final:.6f} GHz,"
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


def measure_ef_chevron_pattern(
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
    Measure an EF chevron pattern for a single qubit.

    The pulse sequence is ``X_ge -> EF drive pulse -> X_ge -> readout``.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    ge_label = exp.ctx.resolve_ge_label(target)
    ef_label = exp.ctx.resolve_ef_label(target)

    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 41)
    if time_range is None:
        time_range = np.linspace(0, 250, 26)
    if frequency is None:
        frequency = exp.ctx.targets[ef_label].frequency
    if amplitude is None:
        amplitude = exp.ctx.params.get_ef_control_amplitude(qubit)
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
    control_sampling_period = exp.ctx.measurement.sampling_period

    def sequence(point_index: int) -> PulseSchedule:
        detuning, duration = sweep_points[int(point_index)]
        ef_drive = FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=0.0,
            sampling_period=control_sampling_period,
        ).detuned(detuning)
        with PulseSchedule() as ps:
            ps.add(ge_label, exp.pulse.x180(ge_label))
            ps.barrier()
            ps.add(ef_label, ef_drive)
            ps.barrier()
            ps.add(ge_label, exp.pulse.x180(ge_label))
        return ps

    with exp.ctx.util.no_output():
        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(sweep_points)),
            frequencies={ef_label: frequency},
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )

    data = sweep_result.data[qubit]
    z = np.asarray(data.data, dtype=np.complex128)

    # shape: (detuning, time)
    raw_data = z.reshape(detuning_range.size, time_range.size)
    chevron_data = _project_chevron_raw_data(raw_data)

    fig = _make_chevron_figure(
        target=ef_label,
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
            "target": ef_label,
            "qubit": qubit,
            "frequency": frequency,
            "amplitude": amplitude,
            "raw_data": raw_data,
            "chevron_data": chevron_data,
        },
        figure=fig,
    )


def measure_fh_chevron_pattern(
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
    Measure an FH chevron pattern for a single qubit.

    The pulse sequence is
    ``X_ge -> X_ef -> FH drive pulse -> X_ef -> X_ge -> readout``.
    """
    qubit = exp.ctx.resolve_qubit_label(target)
    ge_label = exp.ctx.resolve_ge_label(target)
    ef_label = exp.ctx.resolve_ef_label(target)
    fh_label = exp.ctx.resolve_fh_label(target)

    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 41)
    if time_range is None:
        time_range = np.linspace(0, 250, 26)
    if frequency is None:
        frequency = exp.ctx.targets[fh_label].frequency
    if amplitude is None:
        amplitude = exp.ctx.params.control_amplitude[qubit] / (3**0.5)
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
    control_sampling_period = exp.ctx.measurement.sampling_period

    def sequence(point_index: int) -> PulseSchedule:
        detuning, duration = sweep_points[int(point_index)]
        fh_drive = FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=0.0,
            sampling_period=control_sampling_period,
        ).detuned(detuning)
        with PulseSchedule() as ps:
            ps.add(ge_label, exp.pulse.x180(ge_label))
            ps.barrier()
            ps.add(ef_label, exp.pulse.x180(ef_label))
            ps.barrier()
            ps.add(fh_label, fh_drive)
            ps.barrier()
            ps.add(ef_label, exp.pulse.x180(ef_label))
            ps.barrier()
            ps.add(ge_label, exp.pulse.x180(ge_label))
        return ps

    with exp.ctx.util.no_output():
        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=np.arange(len(sweep_points)),
            frequencies={fh_label: frequency},
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )

    data = sweep_result.data[qubit]
    z = np.asarray(data.data, dtype=np.complex128)

    # shape: (detuning, time)
    raw_data = z.reshape(detuning_range.size, time_range.size)
    chevron_data = _project_chevron_raw_data(raw_data)

    fig = _make_chevron_figure(
        target=fh_label,
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
            "target": fh_label,
            "qubit": qubit,
            "frequency": frequency,
            "amplitude": amplitude,
            "raw_data": raw_data,
            "chevron_data": chevron_data,
        },
        figure=fig,
    )


def _measure_and_analyze_fh_chevron(
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
    measurement_result = measure_fh_chevron_pattern(
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
        background_radius=background_radius,
        plot=plot,
        save_image=save_image,
    )

    return measurement_result, analysis_result


def _measure_and_analyze_ef_chevron(
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
    measurement_result = measure_ef_chevron_pattern(
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
        background_radius=background_radius,
        plot=plot,
        save_image=save_image,
    )

    return measurement_result, analysis_result


def _normalize_fh_targets(
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


def _normalize_fh_frequencies(
    exp: Experiment,
    targets: Collection[str],
    frequencies: dict[str, float] | None,
) -> dict[str, float]:
    normalized = {}
    for target in targets:
        fh_label = exp.ctx.resolve_fh_label(target)
        qubit = exp.ctx.resolve_qubit_label(target)
        if frequencies is None:
            normalized[fh_label] = exp.ctx.targets[fh_label].frequency
        elif fh_label in frequencies:
            normalized[fh_label] = frequencies[fh_label]
        else:
            normalized[fh_label] = frequencies[qubit]
    return normalized


def _normalize_ef_frequencies(
    exp: Experiment,
    targets: Collection[str],
    frequencies: dict[str, float] | None,
) -> dict[str, float]:
    normalized = {}
    for target in targets:
        ef_label = exp.ctx.resolve_ef_label(target)
        qubit = exp.ctx.resolve_qubit_label(target)
        if frequencies is None:
            normalized[ef_label] = exp.ctx.targets[ef_label].frequency
        elif ef_label in frequencies:
            normalized[ef_label] = frequencies[ef_label]
        else:
            normalized[ef_label] = frequencies[qubit]
    return normalized


def _normalize_fh_amplitudes(
    exp: Experiment,
    targets: Collection[str],
    amplitudes: dict[str, float] | None,
) -> dict[str, float]:
    normalized = {}
    for target in targets:
        fh_label = exp.ctx.resolve_fh_label(target)
        qubit = exp.ctx.resolve_qubit_label(target)
        if amplitudes is None:
            normalized[fh_label] = exp.ctx.params.control_amplitude[qubit] / (3**0.5)
        elif fh_label in amplitudes:
            normalized[fh_label] = amplitudes[fh_label]
        else:
            normalized[fh_label] = amplitudes[qubit]
    return normalized


def _normalize_ef_amplitudes(
    exp: Experiment,
    targets: Collection[str],
    amplitudes: dict[str, float] | None,
) -> dict[str, float]:
    normalized = {}
    for target in targets:
        ef_label = exp.ctx.resolve_ef_label(target)
        qubit = exp.ctx.resolve_qubit_label(target)
        if amplitudes is None:
            normalized[ef_label] = exp.ctx.params.get_ef_control_amplitude(qubit)
        elif ef_label in amplitudes:
            normalized[ef_label] = amplitudes[ef_label]
        else:
            normalized[ef_label] = amplitudes[qubit]
    return normalized


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
        raw_parts.append(np.asarray(result.data["raw_data"], dtype=np.complex128))

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
            "target": target,
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
