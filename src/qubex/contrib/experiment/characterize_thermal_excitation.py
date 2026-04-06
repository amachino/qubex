from __future__ import annotations

import numpy as np
import qxvisualizer as viz
from qxpulse import FlatTop, PulseSchedule

import qubex.analysis.fitting as fitting
from qubex import Experiment
from qubex.experiment.experiment_result import ExperimentResult, SweepData
from qubex.system.target import Target


def characterize_thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    target: str,
    time_range: np.ndarray,
    ramptime: float | None = None,
) -> float:

    if ramptime is None:
        ramptime = 0

    effective_time_range = time_range + ramptime

    def _sequence_g_population_rabi(T: int) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
            ps.add(target, exp.x180(target))
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

    def _sequence_e_population_rabi(T: int) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
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

    result_g: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_g_population_rabi,
        sweep_range=time_range,
        plot=True,
    )

    result_e: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_e_population_rabi,
        sweep_range=time_range,
        plot=True,
    )

    result_g.plot(normalize=True)
    result_e.plot(normalize=True)

    fit_result_g: fitting.FitResult = fitting.fit_cosine(
        x=effective_time_range,
        y=result_g.data[target].normalized,
        ylabel="Normalized signal",
        plot=True,
    )
    fit_result_e: fitting.FitResult = fitting.fit_cosine(
        x=effective_time_range,
        y=result_e.data[target].normalized,
        xlabel="Time (ns)",
        ylabel="Normalized signal",
        plot=True,
    )

    fig = viz.make_figure()

    fig_g = fit_result_g.data.get("fig")
    if fig_g is not None:
        for trace in fig_g.data:
            trace_name = trace.name
            trace.name = trace_name + " (g population rabi)"
            fig.add_trace(trace)

    fig_e = fit_result_e.data.get("fig")
    if fig_e is not None:
        for trace in fig_e.data:
            trace_name = trace.name
            trace.name = trace_name + " (e population rabi)"
            fig.add_trace(trace)

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
