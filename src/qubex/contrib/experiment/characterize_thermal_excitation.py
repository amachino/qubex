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
) -> float:

    def _sequence_g_population_rabi(T: int) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
            ps.add(target, exp.x180(target))
            ps.barrier()
            ps.add(ef_label, FlatTop(duration=T))
            ps.barrier()
            ps.add(target, exp.x180(target))
        return ps

    def _sequence_e_population_rabi(T: int) -> PulseSchedule:
        ef_label = Target.ef_label(target)

        with PulseSchedule() as ps:
            ps.add(ef_label, FlatTop(duration=T))
            ps.barrier()
            ps.add(target, exp.x180(target))
        return ps

    result_g: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_g_population_rabi,
        sweep_range=time_range,
        plot=False,
    )

    result_e: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_e_population_rabi,
        sweep_range=time_range,
        plot=False,
    )

    state_centers = exp.state_centers[target]
    if state_centers is None:
        raise ValueError(f"State centers for target {target} are not defined.")
    if len(state_centers) < 3:
        raise ValueError(
            f"State centers for target {target} should have at least 3 states (g, e, f)."
        )

    def normalized(data: np.ndarray[np.complex128]) -> np.ndarray:
        c_g = state_centers[0]
        c_f = state_centers[2]

        v_gf = c_f - c_g
        v_gd = data - c_g
        return np.real(v_gd * np.conj(v_gf)) / np.abs(v_gf)

    data_g = normalized(result_g.data[target].data)
    data_e = normalized(result_e.data[target].data)

    fit_result_g: fitting.FitResult = fitting.fit_rabi()
    fit_result_e: fitting.FitResult = fitting.fit_rabi()

    fig = viz.make_figure()
    for d in fit_result_g.data["fig"]:
        for trace in d.data:
            fig.add_trace(trace)
    for d in fit_result_e.data["fig"]:
        for trace in d.data:
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
