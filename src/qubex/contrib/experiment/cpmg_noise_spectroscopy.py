"""
Experimental CPMG noise spectroscopy utilities.

This module provides a frequency-swept CPMG workflow for exploratory noise
spectroscopy. The implementation intentionally lives in ``contrib`` because the
API and the physical interpretation of some edge cases, especially the
``f=0`` limit, remain experimental.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

import qubex as qx
from qubex import visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment.models.experiment_result import (
    T2Data,
)
from qubex.pulse import Blank, PulseSchedule

# Public API exported from this module (contrib-style utilities)
__all__ = [
    "cpmg_noise_spectroscopy",
    "plot_cpmg_results",
]


class CPMGResult(TypedDict):
    """Serialized result bundle used for plotting and on-disk storage."""

    timestamp: str
    targets: list[str]
    n_repeats: int
    frequency_range: list[float]
    tau_spacing_list_ns: list[int]
    time_range: list[float]
    data: np.ndarray


def plot_cpmg_results(
    ex: qx.Experiment,
    result: CPMGResult | None = None,
    save_dir: str | Path | None = None,
    mode: Literal["t2", "gamma2"] = "t2",
    show_T1: bool = True,
    plot: bool = True,
    save_fig: bool = True,
) -> go.Figure:
    """
    Plot aggregated CPMG spectroscopy results.

    Parameters
    ----------
    ex
        Experiment object used to load auxiliary calibration data such as T1.
    result
        In-memory result bundle returned by :func:`cpmg_noise_spectroscopy`.
    save_dir
        Directory containing ``params.json`` and ``t2_matrix.npy``.
    mode
        Plot ``"t2"`` or ``"gamma2"``.
    show_T1
        Overlay ``2*T1`` references when plotting ``T2``.
    plot
        Display the figure interactively.
    save_fig
        Save the figure to disk.
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if mode not in ["t2", "gamma2"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 't2' or 'gamma2'.")
    if mode == "gamma2" and show_T1:
        print(
            "Warning: show_T1 is not applicable when mode is 'gamma2'. Ignoring show_T1."
        )

    if result is not None:
        if save_dir is not None:
            print(
                "Warning: Both result and save_dir provided. Ignoring save_dir and using result for plotting."
            )
        targets = result["targets"]
        n_repeats = result["n_repeats"]
        frequency_range = np.array(result["frequency_range"])
        t2_matrix = result["data"]
        timestamp = result["timestamp"]
    elif save_dir is not None:
        params_path = save_dir / Path("params.json")
        with params_path.open("r", encoding="utf-8") as fp:
            params = json.load(fp)

        targets = params["targets"]
        n_repeats = params["n_repeats"]
        timestamp = params["timestamp"]
        frequency_range = np.array(params["frequency_range"])
        t2_matrix = np.load(save_dir / "t2_matrix.npy")
    else:
        raise ValueError("Either result or save_dir must be provided to plot results.")

    fig = viz.make_figure()
    for i, target in enumerate(targets):
        t2_ave = np.nanmean(t2_matrix[i], axis=0)
        if mode == "t2":
            if show_T1:
                t1_dict = ex.ctx.system_manager.config_loader.load_param_data("t1")
                fig.add_hline(
                    y=2 * t1_dict[target] * 1e-3,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"2*T1 of {target}",
                    annotation_position="top left",
                )
            fig.add_trace(
                go.Scatter(
                    x=frequency_range * 1e3,
                    y=t2_ave * 1e-3,
                    mode="markers+lines",
                    name=target,
                )
            )
            yaxis_title = "Average T2_CPMG (us)"
        else:  # mode == "gamma2"
            fig.add_trace(
                go.Scatter(
                    x=frequency_range * 1e3,
                    y=1 / (t2_ave * 1e-3),
                    mode="markers+lines",
                    name=f"{target} (1/T2)",
                )
            )
            yaxis_title = "Average Gamma_2_CPMG (1/us)"

    yaxis_layout = dict(title=yaxis_title, showgrid=True)
    if mode == "t2":
        yaxis_layout["dtick"] = 10

    fig.update_layout(
        title=dict(
            text="CPMG Noise Spectroscopy",
            subtitle=dict(
                text=f"timestamp: {timestamp}, repetition: {n_repeats} times"
            ),
        ),
        xaxis=dict(
            title="Effective CPMG Frequency (MHz)", tick0=0, dtick=10, showgrid=True
        ),
        yaxis=yaxis_layout,
        xaxis_type="linear",
        yaxis_type="linear",
        width=800,
        height=600,
    )
    if save_fig:
        if save_dir is None:
            save_dir = Path("cpmg_noise_spectroscopy") / Path(timestamp)
        fig.write_image(save_dir / Path("cpmg_noise_spectroscopy_summary.png"))
        print(
            f"Saved CPMG summary figure to {save_dir / 'cpmg_noise_spectroscopy_summary.png'}"
        )
    if plot:
        fig.show()

    return fig


def cpmg_noise_spectroscopy(
    ex: qx.Experiment,
    frequency_range: ArrayLike,
    time_range: ArrayLike,
    targets: list[str] | str | None = None,
    pi_pulse_pattern: Literal["++++", "+--+", "+-+-"] = "++++",
    plot_mode: Literal["t2", "gamma2"] = "t2",
    shot_interval: float | None = None,
    save_t2_image: bool = False,
    save_cpmg_data: bool = True,
    save_cpmg_fig: bool = True,
    n_repeats: int = 10,
    n_shots: int = 1024,
    save_dir: str | Path = "cpmg_noise_spectroscopy",
    plot: bool = False,
) -> tuple[np.ndarray, go.Figure]:
    """
    Run a CPMG noise spectroscopy experiment sweeping frequency and total time.

    This routine accepts a sweep of effective CPMG frequencies in GHz and desired
    total sequence durations in ns. For each effective frequency, it constructs a
    CPMG sequence whose pi-count ``n`` is chosen so that the total sequence time
    is closest to each requested element of ``time_range``.

    Notes
    -----
    - The hardware timing grid is assumed to be 2 ns.
    - Input frequencies are converted to half-tau values and rounded to that grid.
      Distinct input frequencies may therefore collapse onto the same realized
      half-tau and be merged.
    - The ``f=0`` point is treated as a no-pi reference sequence, corresponding to
      the experimental ``tau -> infinity`` limit rather than a finite-frequency
      CPMG filter point.

    Parameters
    ----------
    ex
        Experiment object providing pulse definitions and the sweep API.
    frequency_range
        Sweep of effective CPMG frequencies in GHz. All values must be finite and
        non-negative.
    time_range
        Target total sequence durations in ns. All values must be finite and
        non-negative.
    targets
        Target qubit label(s). If ``None``, all labels in ``ex.qubit_labels`` are used.
    pi_pulse_pattern
        Sign pattern for the CPMG pi pulses.
    plot_mode
        Summary plot mode, either ``"t2"`` or ``"gamma2"``.
    shot_interval
        Delay between repeated shots passed through to the measurement backend.
    save_t2_image
        Save per-frequency T2 fit images.
    save_cpmg_data
        Save ``params.json`` and ``t2_matrix.npy``.
    save_cpmg_fig
        Save the summary figure.
    n_repeats
        Number of repeated measurements for each target and frequency.
    n_shots
        Number of shots per sweep point.
    save_dir
        Base directory for saved outputs.
    plot
        Display the summary figure.

    Returns
    -------
    tuple[np.ndarray, go.Figure]
        ``(t2_matrix, fig)`` where ``t2_matrix`` has shape
        ``(n_targets, n_repeats, n_frequencies)`` and stores fitted ``T2`` in ns.
    """
    if not targets:
        raise ValueError("targets must contain at least one target.")
    if targets is None:
        targets = list(ex.qubit_labels)
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    time_range = np.asarray(time_range, dtype=float)
    if time_range.ndim != 1 or time_range.size == 0:
        raise ValueError("time_range must be a non-empty 1D array.")
    if not np.all(np.isfinite(time_range)):
        raise ValueError("time_range must contain only finite values.")
    if np.any(time_range < 0):
        raise ValueError("time_range must contain only non-negative values.")
    if n_repeats <= 0:
        raise ValueError(f"n_repeats must be positive, got {n_repeats}.")
    if n_shots <= 0:
        raise ValueError(f"n_shots must be positive, got {n_shots}.")

    frequency_range, half_tau_list = _cpmg_effective_frequencies(frequency_range)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir)
    base_dir = save_dir / Path(timestamp)
    if save_t2_image or save_cpmg_data or save_cpmg_fig:
        base_dir.mkdir(parents=True, exist_ok=True)

    def get_pi_duration(targets: list[str]) -> int:
        pi_durations = np.array(
            [int(ex.hpi_pulse[target].duration * 2) for target in targets]
        )
        if pi_durations.size == 0:
            raise ValueError(f"No valid targets provided: {targets}")
        if np.isnan(pi_durations).any():
            raise ValueError(
                f"Invalid pi pulse duration for targets {targets}: {pi_durations}"
            )
        if not np.all(pi_durations == pi_durations[0]):
            raise ValueError(
                f"Inconsistent pi pulse durations for targets {targets}: {pi_durations}"
            )
        return pi_durations[0]

    def build_cpmg_sequence(
        target: str,
        half_tau: int,
        n: int,
        pattern: Literal["++++", "+--+", "+-+-"] = "++++",
    ) -> PulseSchedule:
        if target not in ex.hpi_pulse:
            raise ValueError(f"Target {target} not found in exp.hpi_pulse")
        if n <= 0:
            raise ValueError(f"Number of pi pulses n must be positive, got {n}")
        if half_tau <= 0 or half_tau % 2 != 0:
            raise ValueError(
                f"Half tau must be a positive even integer, got {half_tau}"
            )

        with PulseSchedule() as ps:
            ps.add(target, ex.hpi_pulse[target])
            pi_pulse = ex.x180(target).shifted(np.pi / 2)
            for i in range(n):
                ps.add(target, Blank(half_tau))

                if pattern == "++++":
                    ps.add(target, pi_pulse)
                elif pattern == "+--+":
                    if i % 4 in [0, 3]:
                        ps.add(target, pi_pulse)
                    else:
                        ps.add(target, pi_pulse.scaled(-1))
                elif pattern == "+-+-":
                    if i % 4 in [0, 2]:
                        ps.add(target, pi_pulse)
                    else:
                        ps.add(target, pi_pulse.scaled(-1))
                else:
                    raise ValueError(
                        f"Invalid pattern: {pattern}. Must be '++++', '+--+', or '+-+-'."
                    )
                ps.add(target, Blank(half_tau))

            ps.add(target, ex.hpi_pulse[target].scaled(-1))
        return ps

    def build_no_pi_sequence(
        target: str,
        tau: int,
    ) -> PulseSchedule:
        if target not in ex.hpi_pulse:
            raise ValueError(f"Target {target} not found in exp.hpi_pulse")
        if tau < 0:
            raise ValueError(f"Evolution time must be non-negative, got {tau}")

        with PulseSchedule() as ps:
            ps.add(target, ex.hpi_pulse[target])
            if tau > 0:
                ps.add(target, Blank(tau))
            ps.add(target, ex.hpi_pulse[target].scaled(-1))
        return ps

    t2_results: dict[str, list[list[T2Data]]] = {target: [] for target in targets}
    for _rep in tqdm(range(n_repeats), desc="repetitions"):
        for target in tqdm(targets, leave=False, desc="targets"):
            pi_duration = get_pi_duration([target])
            results_for_qubit: list[T2Data] = []
            for i, half_tau in enumerate(
                tqdm(half_tau_list, leave=False, desc="frequencies")
            ):
                if half_tau == 0:
                    # f=0 case: do not apply any pi pulse between the two hpi pulses.
                    actual_time_range = (time_range // 2) * 2
                    result = ex.sweep_parameter(
                        sequence=lambda tau, target=target: build_no_pi_sequence(
                            target=target, tau=tau
                        ),
                        sweep_range=actual_time_range,
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                    )
                else:
                    # For each desired total time, compute the integer n that makes
                    # T_total = n * (2*half_tau + pi_duration) closest to the target.
                    unit_time = 2 * half_tau + pi_duration
                    n_values = []
                    for T_des in time_range:
                        n_calc = max(1, round(float(T_des) / unit_time))
                        n_values.append(n_calc)

                    n_range = sorted(np.unique(n_values))
                    actual_time_range = np.array([n * unit_time for n in n_range])
                    result = ex.sweep_parameter(
                        sequence=lambda n, target=target, half_tau=half_tau: (
                            build_cpmg_sequence(
                                target=target,
                                half_tau=half_tau,
                                n=n,
                                pattern=pi_pulse_pattern,
                            )
                        ),
                        sweep_range=n_range,
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                    )

                sweep_data = result.data[target]
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=actual_time_range,
                    y=0.5 * (1 + sweep_data.normalized),
                    plot=False,
                    title="T2 echo",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                    xaxis_type="log",
                    yaxis_type="linear",
                )

                _sweep_data = copy.deepcopy(sweep_data)
                _sweep_data.sweep_range = actual_time_range

                if fit_result.status is FitStatus.SUCCESS:
                    t2 = fit_result["tau"]
                    t2_err = fit_result["tau_err"]
                    r2 = fit_result["r2"]

                    t2_data = T2Data.new(
                        _sweep_data,
                        t2=t2,
                        t2_err=t2_err,
                        r2=r2,
                    )

                    if save_t2_image:
                        save_path = base_dir / Path(target)
                        save_path.mkdir(parents=True, exist_ok=True)
                        fit_result.get_figure().write_image(
                            save_path
                            / f"freq{frequency_range[i] * 1e6:.2f}_rep{_rep + 1}.png"
                        )

                    results_for_qubit.append(t2_data)
                else:
                    print(
                        f"Fit failed for target {target}, frequency {frequency_range[i] * 1e6:.2f} kHz"
                    )
                    t2_data = T2Data.new(
                        _sweep_data,
                        t2=np.nan,
                        t2_err=np.nan,
                        r2=np.nan,
                    )
                    results_for_qubit.append(t2_data)

            t2_results[target].append(results_for_qubit)

    t2_matrix = np.zeros((len(targets), n_repeats, len(half_tau_list)))
    for i, target in enumerate(targets):
        for j in range(n_repeats):
            for k, _half_tau in enumerate(half_tau_list):
                t2_matrix[i, j, k] = t2_results[target][j][k].t2

    params: CPMGResult = {
        "timestamp": timestamp,
        "targets": targets,
        "n_repeats": n_repeats,
        "frequency_range": frequency_range.tolist(),
        "tau_spacing_list_ns": half_tau_list.tolist(),
        "time_range": time_range.tolist(),
        "data": t2_matrix,
    }
    if save_cpmg_data:
        params_path = base_dir / Path("params.json")
        serializable_params = {
            "timestamp": timestamp,
            "targets": targets,
            "n_repeats": n_repeats,
            "frequency_range": frequency_range.tolist(),
            "tau_spacing_list_ns": half_tau_list.tolist(),
            "time_range": time_range.tolist(),
        }
        with params_path.open("w", encoding="utf-8") as fp:
            json.dump(serializable_params, fp, indent=2)

        np.save(base_dir / "t2_matrix.npy", t2_matrix)
        print(f"Saved T2 results matrix to {base_dir / 't2_matrix.npy'}")

    fig = plot_cpmg_results(
        ex=ex,
        result=params,
        mode=plot_mode,
        plot=plot,
        save_fig=save_cpmg_fig,
        save_dir=base_dir,
    )

    return t2_matrix, fig


def _cpmg_effective_frequencies(
    frequency_range: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert requested effective frequencies into realized half-tau values.

    Parameters
    ----------
    frequency_range
        Effective CPMG frequencies in GHz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Realized effective frequencies in GHz and corresponding half-tau values
        in ns.
    """
    frequency_range = np.asarray(frequency_range, dtype=float)
    if frequency_range.ndim != 1 or frequency_range.size == 0:
        raise ValueError("frequency_range must be a non-empty 1D array.")
    if not np.all(np.isfinite(frequency_range)):
        raise ValueError("frequency_range must contain only finite values.")
    if np.any(frequency_range < 0):
        raise ValueError("frequency_range must contain only non-negative values.")

    half_tau_list = []
    non_0_idx = np.where(frequency_range != 0)[0]
    for _i, f in enumerate(frequency_range[non_0_idx]):
        # Convert f (GHz) to half-tau (ns) and snap to the 2 ns hardware grid.
        half_tau_ns = 1 / (4 * float(f))
        half_tau_rounded = int(max(2, int(round(half_tau_ns / 2.0) * 2)))
        half_tau_list.append(half_tau_rounded)

    half_tau_list = np.array(half_tau_list, dtype=int)
    half_tau_list = np.sort(np.unique(half_tau_list))[::-1]
    _frequency_range = 0.25 / half_tau_list

    if non_0_idx.size != frequency_range.size:
        _frequency_range = np.concatenate(([0], _frequency_range))
        half_tau_list = np.concatenate(([0], half_tau_list))

    return _frequency_range, half_tau_list
