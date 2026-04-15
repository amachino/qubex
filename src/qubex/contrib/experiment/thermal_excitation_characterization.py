from __future__ import annotations

from collections import defaultdict

import numpy as np
from qxpulse import FlatTop, PulseSchedule
from tqdm import tqdm

import qubex.analysis.fitting as fitting
from qubex import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_SHOTS,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models.experiment_result import ExperimentResult, SweepData
from qubex.experiment.models.result import Result
from qubex.system.target import Target


def thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    amplitude_range: np.ndarray | None = None,
    time_range: np.ndarray | None = None,
    n_amplitude_ranges: int | None = None,
    ef_rabi_ramptime: float | None = None,
    ef_rabi_amplitude: float | None = None,
    n_shots: int = DEFAULT_SHOTS,
    plot: bool = False,
) -> float:
    """
    Estimate the thermal excitation probability of a qubit via ef Rabi oscillations.

    For each drive amplitude in `amplitude_range`, a flat-top preparation pulse is
    applied to the ge transition, followed by an ef Rabi sweep (|e⟩–|f⟩ transition),
    and finally a ge π pulse.  The resulting Rabi oscillation amplitude varies with
    the preparation amplitude according to the thermal population in |e⟩.  Fitting a
    cosine to those amplitudes yields the minimum (`rabi_ampl_min`) and maximum
    (`rabi_ampl_max`) Rabi amplitudes, from which the thermal excitation probability
    is computed as

        p_ex = rabi_ampl_min / (rabi_ampl_min + rabi_ampl_max).

    Parameters
    ----------
    exp : Experiment
        Experiment instance used to run pulse sequences and access calibration
        parameters.
    target : str
        Label of the qubit target to characterize.
    amplitude_range : np.ndarray, optional
        Preparation pulse amplitudes to sweep (a.u.).  If `None`, a linearly spaced
        array of `n_amplitude_ranges` points from 0 to 1.5× the ge flat-top amplitude
        is used.
    time_range : np.ndarray, optional
        ef Rabi pulse durations to sweep (ns).  Defaults to
        `DEFAULT_RABI_TIME_RANGE`.
    n_amplitude_ranges : int, optional
        Number of amplitude points when `amplitude_range` is `None`.  Defaults to
        21.
    ef_rabi_ramptime : float, optional
        Ramp time of the EF Rabi flat-top pulse (ns).  Defaults to 0.
    ef_rabi_amplitude : float, optional
        Drive amplitude for the EF Rabi pulse (a.u.).  Defaults to the ge control
        amplitude divided by √2.
    n_shots : int, optional
        Number of measurement shots per sequence.  Defaults to `DEFAULT_SHOTS`.
    plot : bool, optional
        If `True`, intermediate sweep and fit results are plotted.  Defaults to
        `False`.

    Returns
    -------
    Result
        A `Result` object whose `data` dict contains:

        - `time_range` : np.ndarray – ef Rabi time sweep used.
        - `amplitude_range` : np.ndarray – Preparation amplitude sweep used.
        - `result_history` : list[ExperimentResult] – Raw sweep results for each
          amplitude point.
        - `p_ex` : float – Estimated thermal excitation probability.
        - `rabi_ampl_min` : float – Minimum Rabi oscillation amplitude (a.u.).
        - `rabi_ampl_max` : float – Maximum Rabi oscillation amplitude (a.u.).

        The `figure` attribute holds the cosine-fit plot with amplitude annotations.

    Notes
    -----
    Only EF Rabi fits with R² ≥ 0.9 are included when building the cosine fit.
    This function prints fit parameters and the estimated `p_ex` to stdout.

    """
    if n_amplitude_ranges is None:
        n_amplitude_ranges = 21
    if amplitude_range is None:
        pi_rabi_freq = 1 / (PI_DURATION + PI_RAMPTIME)
        pi_rabi_amplitude = exp.calc_control_amplitude(target, pi_rabi_freq)
        amplitude_range = np.linspace(0, pi_rabi_amplitude * 1.5, n_amplitude_ranges)

    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE

    if ef_rabi_ramptime is None:
        ef_rabi_ramptime = 0

    if ef_rabi_amplitude is None:
        ef_rabi_amplitude = exp.params.control_amplitude[target] / np.sqrt(2)

    time_range = np.asarray(time_range)
    effective_time_range = time_range + ef_rabi_ramptime

    fit_amplitude_history = defaultdict(list)
    fit_rabi_amplitude_history = defaultdict(list)
    result_history = []
    for amplitude in tqdm(amplitude_range):

        def _sequence_population_rabi(
            T: int,
        ) -> PulseSchedule:
            ef_label = Target.ef_label(target)

            with PulseSchedule() as ps:
                ampl_pulse = FlatTop(
                    duration=PI_DURATION,
                    amplitude=amplitude,
                    tau=PI_RAMPTIME,
                )
                ps.add(target, ampl_pulse)
                ps.barrier()
                ps.add(
                    ef_label,
                    FlatTop(
                        duration=T + 2 * ef_rabi_ramptime,
                        amplitude=ef_rabi_amplitude,
                        tau=ef_rabi_ramptime,
                    ),
                )
                ps.barrier()
                ps.add(target, exp.x180(target))
            return ps

        result: ExperimentResult[SweepData] = exp.sweep_parameter(
            sequence=_sequence_population_rabi,
            sweep_range=time_range,
            n_shots=n_shots,
            plot=plot,
        )
        result_history.append(result)

        for target, data in result.data.items():
            fit_rabi_result = fitting.fit_rabi(
                target=target,
                times=effective_time_range,
                data=data.data,
                plot=plot,
            )
            r2 = fit_rabi_result.data["r2"]
            if r2 >= 0.9:
                fit_rabi_amplitude_history[target].append(
                    fit_rabi_result.data["amplitude"]
                )
                fit_amplitude_history[target].append(amplitude)

    for target in fit_amplitude_history.keys():
        fit_cosine_result = fitting.fit_cosine(
            x=fit_amplitude_history[target],
            y=fit_rabi_amplitude_history[target],
            plot=False,
        )

        popt = fit_cosine_result.data["popt"]
        densex = np.linspace(0, amplitude_range[-1], 1000)
        y_fit = fitting.func_cos(densex, *popt)
        idx_min = int(np.argmin(y_fit))
        idx_max = int(np.argmax(y_fit))
        x_min = densex[idx_min]
        x_max = densex[idx_max]
        rabi_ampl_min = np.min(y_fit)
        rabi_ampl_max = np.max(y_fit)
        p_ex = rabi_ampl_min / (rabi_ampl_min + rabi_ampl_max)
        ef_rabi_freq = exp.calc_control_amplitude(target, ef_rabi_amplitude)

        fig = fit_cosine_result.figure
        fig.data = tuple(trace for trace in fig.data if trace.name != "Fit")
        fig.add_scatter(
            x=densex,
            y=y_fit,
            mode="lines",
            name="Fit Extrapolation",
        )
        # Move the fit trace to the end of the data list to ensure it is plotted on top
        fig.data = (
            fig.data[-1],
            *fig.data[:-1],
        )
        fig.update_layout(
            title=dict(
                text=f"Thermal excitation characterization via Rabi - {target}",
                subtitle=dict(
                    text=f"Ωef = {ef_rabi_freq * 1e3:.1f} MHz, p_ex = {p_ex:.4f}"
                ),
            ),
            xaxis_title="Amplitude (a.u.)",
            yaxis_title="Rabi Amplitude (a.u.)",
        )
        fig.add_annotation(
            x=x_min,
            y=rabi_ampl_min,
            text=f"min: {rabi_ampl_min:.6g}",
            showarrow=True,
            arrowhead=1,
        )
        fig.add_annotation(
            x=x_max,
            y=rabi_ampl_max,
            text=f"max: {rabi_ampl_max:.6g}",
            showarrow=True,
            arrowhead=1,
        )

        fig.show()

        A = fit_cosine_result.data["A"]
        A_err = fit_cosine_result.data["A_err"]
        f = fit_cosine_result.data["f"]
        f_err = fit_cosine_result.data["f_err"]
        phi = fit_cosine_result.data["phi"]
        phi_err = fit_cosine_result.data["phi_err"]
        C = fit_cosine_result.data["C"]
        C_err = fit_cosine_result.data["C_err"]

        print("")
        print(f"target : {target}")
        print(f"A   : {A} ± {A_err}")
        print(f"f   : {f} ± {f_err}")
        print(f"phi : {phi} ± {phi_err}")
        print(f"C   : {C} ± {C_err}")
        print("")
        print("thermal excitation probability (p_ex):")
        print(f"rabi amplitude min : {rabi_ampl_min:.4f}")
        print(f"rabi amplitude max : {rabi_ampl_max:.4f}")
        print(f"p_ex : {p_ex:.4f}")
        print("")
    return Result(
        data={
            "time_range": time_range,
            "amplitude_range": amplitude_range,
            "result_history": result_history,
            "p_ex": p_ex,
            "rabi_ampl_min": rabi_ampl_min,
            "rabi_ampl_max": rabi_ampl_max,
        },
        figure=fig,
    )
