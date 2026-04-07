from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
from qxpulse import FlatTop, PulseSchedule
from tqdm import tqdm

import qubex.analysis.fitting as fitting
from qubex import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_RABI_TIME_RANGE,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.experiment_result import ExperimentResult, SweepData
from qubex.experiment.result import Result
from qubex.pulse import VirtualZ
from qubex.system.target import Target


def characterize_thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    sweep_range: np.ndarray | None = None,
    time_range: np.ndarray | None = None,
    ef_rabi_ramptime: float | None = None,
    ef_rabi_amplitude: float | None = None,
    plot: bool = False,
    sweep_mode: Literal["amplitude", "virtual-Z"] = "amplitude",
) -> float:
    if sweep_mode == "amplitude":
        return _characterize_thermal_excitation_via_sweep_amplitude(
            exp=exp,
            target=target,
            time_range=time_range,
            amplitude_range=sweep_range,
            ef_rabi_amplitude=ef_rabi_amplitude,
            ef_rabi_ramptime=ef_rabi_ramptime,
            plot=plot,
        )
    elif sweep_mode == "virtual-Z":
        return _characterize_thermal_excitation_via_virtual_z(
            exp=exp,
            target=target,
            time_range=time_range,
            theta_range=sweep_range,
            ef_rabi_amplitude=ef_rabi_amplitude,
            ef_rabi_ramptime=ef_rabi_ramptime,
            plot=plot,
        )


def _characterize_thermal_excitation_via_sweep_amplitude(
    exp: Experiment,
    *,
    target: str,
    amplitude_range: np.ndarray | None = None,
    time_range: np.ndarray | None = None,
    ef_rabi_ramptime: float | None = None,
    ef_rabi_amplitude: float | None = None,
    plot: bool = False,
) -> float:

    if amplitude_range is None:
        pi_rabi_freq = 1 / (PI_DURATION + PI_RAMPTIME)
        pi_rabi_amplitude = exp.calc_control_amplitude(target, pi_rabi_freq)
        amplitude_range = np.linspace(0, pi_rabi_amplitude * 1.5, 21)

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

    fit_cosine_result = defaultdict(list)
    for target in fit_amplitude_history.keys():
        fit_cosine_result[target] = fitting.fit_cosine(
            x=fit_amplitude_history[target],
            y=fit_rabi_amplitude_history[target],
            plot=False,
        )

        popt = fit_cosine_result[target].data["popt"]
        densex = np.linspace(0, amplitude_range[-1], 1000)
        y_fit = fitting.func_cos(densex, *popt)
        rabi_ampl_min = np.min(y_fit)
        rabi_ampl_max = np.max(y_fit)
        p_ex = rabi_ampl_min / (rabi_ampl_min + rabi_ampl_max)
        ef_rabi_freq = exp.calc_control_amplitude(target, ef_rabi_amplitude)

        fig = fit_cosine_result[target].data["fig"]
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

        fig.show()

        A = fit_cosine_result[target].data["A"]
        A_err = fit_cosine_result[target].data["A_err"]
        f = fit_cosine_result[target].data["f"]
        f_err = fit_cosine_result[target].data["f_err"]
        phi = fit_cosine_result[target].data["phi"]
        phi_err = fit_cosine_result[target].data["phi_err"]
        C = fit_cosine_result[target].data["C"]
        C_err = fit_cosine_result[target].data["C_err"]

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


def _characterize_thermal_excitation_via_virtual_z(
    exp: Experiment,
    *,
    target: str,
    time_range: np.ndarray | None = None,
    theta_range: np.ndarray | None = None,
    ef_rabi_ramptime: float | None = None,
    ef_rabi_amplitude: float | None = None,
    plot: bool = False,
) -> float:

    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if theta_range is None:
        theta_range = np.linspace(0, 1.5 * np.pi, 21)

    if ef_rabi_ramptime is None:
        ef_rabi_ramptime = 0

    if ef_rabi_amplitude is None:
        ef_rabi_amplitude = exp.params.control_amplitude[target] / np.sqrt(2)

    time_range = np.asarray(time_range)
    effective_time_range = time_range + ef_rabi_ramptime

    fit_theta_history = defaultdict(list)
    fit_rabi_amplitude_history = defaultdict(list)
    result_history = []
    for theta in tqdm(theta_range):

        def _sequence_population_rabi(
            T: int,
        ) -> PulseSchedule:
            ef_label = Target.ef_label(target)

            with PulseSchedule() as ps:
                ps.add(target, exp.y90(target))
                ps.add(target, VirtualZ(theta))
                ps.add(target, exp.y90m(target))
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
                fit_theta_history[target].append(theta)

    fit_cosine_result = defaultdict(list)
    for target in fit_theta_history.keys():
        fit_cosine_result[target] = fitting.fit_cosine(
            x=fit_theta_history[target],
            y=fit_rabi_amplitude_history[target],
            plot=False,
        )

        popt = fit_cosine_result[target].data["popt"]
        densex = np.linspace(0, theta_range[-1], 1000)
        y_fit = fitting.func_cos(densex, *popt)
        rabi_ampl_min = np.min(y_fit)
        rabi_ampl_max = np.max(y_fit)
        p_ex = rabi_ampl_min / (rabi_ampl_min + rabi_ampl_max)
        ef_rabi_freq = exp.calc_control_amplitude(target, ef_rabi_amplitude)

        fig = fit_cosine_result[target].data["fig"]
        fig.update_layout(
            title=dict(
                text=f"Thermal excitation characterization via Rabi - {target}",
                subtitle=dict(
                    text=f"Ωef = {ef_rabi_freq * 1e3:.1f} MHz, p_ex = {p_ex:.4f}"
                ),
            ),
            xaxis_title="Theta (rad)",
            yaxis_title="Rabi Amplitude (a.u.)",
        )

        fig.show()

        A = fit_cosine_result[target].data["A"]
        A_err = fit_cosine_result[target].data["A_err"]
        f = fit_cosine_result[target].data["f"]
        f_err = fit_cosine_result[target].data["f_err"]
        phi = fit_cosine_result[target].data["phi"]
        phi_err = fit_cosine_result[target].data["phi_err"]
        C = fit_cosine_result[target].data["C"]
        C_err = fit_cosine_result[target].data["C_err"]

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
            "theta_range": theta_range,
            "result_history": result_history,
            "p_ex": p_ex,
            "rabi_ampl_min": rabi_ampl_min,
            "rabi_ampl_max": rabi_ampl_max,
        },
        figure=fig,
    )


def characterize_thermal_excitation_via_Gaussian_fit(
    exp: Experiment,
    *,
    targets: list[str],
    n_shots: int = 1024,
) -> float:

    for target in targets:
        results = exp.measure_state_distribution(
            target,
            n_shots=n_shots,
        )
        iq_g = results[0].data[target].kerneled
        iq_e = results[1].data[target].kerneled


# TODO: Fit Gaussian distributions to iq_g and iq_e
