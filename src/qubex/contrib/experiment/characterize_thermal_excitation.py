from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from qxpulse import FlatTop, PulseSchedule

import qubex.analysis.fitting as fitting
from qubex import Experiment
from qubex.experiment.experiment_result import ExperimentResult, SweepData
from qubex.pulse import VirtualZ
from qubex.system.target import Target


def characterize_thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    time_range: np.ndarray,
    ramptime: float | None = None,
    obtain_reference_gf: bool = True,
) -> float:

    if ramptime is None:
        ramptime = 0

    effective_time_range = time_range + ramptime
    reference_points = exp.obtain_reference_points(
        target,
    )

    theta_list = np.linspace(0, np.pi, 11)

    fit_rabi_amplitudes = defaultdict(list)
    for theta in theta_list:

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
                        duration=T,
                        amplitude=exp.params.control_amplitude[target] / np.sqrt(2),
                        tau=ramptime,
                    ),
                )
                ps.barrier()
                ps.add(target, exp.x180(target))
            return ps

        result: ExperimentResult[SweepData] = exp.sweep_parameter(
            sequence=_sequence_population_rabi,
            sweep_range=time_range,
            plot=True,
        )

        for target, data in result.data.items():
            fit_result = fitting.fit_rabi(
                target=target,
                times=effective_time_range,
                data=result.data[target].data,
                reference_point=reference_points.data["iq"][target],
                plot=True,
            )
            fit_rabi_amplitudes[target].append(fit_result.data["amplitude"])

    fig = viz.make_figure()

    fig.add_trace(
        go.Scatter(
            x=theta_list,
            y=fit_rabi_amplitudes[target],
            mode="markers+lines",
            name="Rabi amplitude",
        )
    )
    fig.update_xaxes(
        title_text="θ (rad)",
    )
    fig.update_yaxes(
        title_text="Rabi amplitude (a.u.)",
    )
    fig.show()
    print("")
    print(f"target : {target}")
    print("p_ex: xx")
    print("")


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
