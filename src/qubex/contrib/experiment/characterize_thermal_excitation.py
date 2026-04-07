from __future__ import annotations

from collections import defaultdict

import numpy as np
from qxpulse import FlatTop, PulseSchedule
from tqdm import tqdm

import qubex.analysis.fitting as fitting
from qubex import Experiment
from qubex.experiment.experiment_constants import DEFAULT_RABI_TIME_RANGE
from qubex.experiment.experiment_result import ExperimentResult, SweepData
from qubex.experiment.result import Result
from qubex.pulse import VirtualZ
from qubex.system.target import Target


def characterize_thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    time_range: np.ndarray | None = None,
    theta_range: np.ndarray | None = None,
    ramptime: float | None = None,
    ef_rabi_amplitude: float | None = None,
    plot: bool = False,
) -> float:

    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if theta_range is None:
        theta_range = np.linspace(0, 1.5 * np.pi, 21)

    if ramptime is None:
        ramptime = 0

    reference_points = exp.obtain_reference_points(
        target,
    )
    if ef_rabi_amplitude is None:
        ef_rabi_amplitude = exp.params.control_amplitude[target] / np.sqrt(2)

    time_range = np.asarray(time_range)
    effective_time_range = time_range + ramptime

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
                        duration=T + 2 * ramptime,
                        amplitude=ef_rabi_amplitude,
                        tau=ramptime,
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
                data=result.data[target].data,
                reference_point=reference_points.data["iq"][target],
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
        rabi_ampl_0 = fitting.func_cos(0, *popt)
        rabi_ampl_pi = fitting.func_cos(np.pi, *popt)
        p_ex = rabi_ampl_0 / (rabi_ampl_0 + rabi_ampl_pi)
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
        print(f"rabi amplitude at θ=0 : {rabi_ampl_0:.4f}")
        print(f"rabi amplitude at θ=π : {rabi_ampl_pi:.4f}")
        print(f"p_ex : {p_ex:.4f}")
        print("")
    return Result(
        data={
            "time_range": time_range,
            "theta_range": theta_range,
            "result_history": result_history,
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
