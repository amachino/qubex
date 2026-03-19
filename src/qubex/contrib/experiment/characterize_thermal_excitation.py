import numpy as np
from qxpulse import PulseSchedule

from qubex import Experiment
from qubex.experiment.experiment_result import ExperimentResult, SweepData


def characterize_thermal_excitation_via_rabi(
    exp: Experiment,
    *,
    targets: list[str],
    time_range: np.ndarray,
) -> float:

    def _sequence_1(T: int) -> PulseSchedule:
        pass

    def _sequence_2(T: int) -> PulseSchedule:
        pass

    result_1: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_1,
        sweep_range=time_range,
        plot=False,
    )

    result_2: ExperimentResult[SweepData] = exp.sweep_parameter(
        sequence=_sequence_2,
        sweep_range=time_range,
        plot=False,
    )

    result_1.plot()
    result_2.plot()


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
