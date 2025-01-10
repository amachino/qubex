from ..system.system import System
from ..interface.iexperiment import IExperiment
from numpy.typing import ArrayLike
from ...pulse import (
    PulseSchedule,
    Rect,
)
import numpy as np
from ...analysis import fitting
from ...experiment.experiment_result import (
    ExperimentResult,
    RabiData,
)
from ...measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)

from ..sweeper.sweeper import Sweeper

class Rabi(IExperiment):
	experiment_name = "Rabi"
	input_parameters = ["qubit", "drive_amplitude", "drive_duration"]
	output_parameters = ["data"]

	def __init__(self,
        amplitudes: dict[str, float],
        time_range: ArrayLike = range(0, 201, 4),
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,):
		self.amplitudes = amplitudes
		self.time_range = np.array(time_range, dtype=np.float64)
		self.frequencies = frequencies
		self.detuning = detuning
		self.shots = shots
		self.interval = interval
		self.plot = plot
		self.store_params = store_params

	def take_data(self,system:System) -> object:
        # target labels
		targets = list(self.amplitudes.keys())
        # rabi sequence with rect pulses of duration T
		def rabi_sequence(T: int) -> PulseSchedule:
			with PulseSchedule(targets) as ps:
				for target in targets:
					ps.add(target, Rect(duration=T, amplitude=self.amplitudes[target]))
			return ps

        # run the Rabi experiment by sweeping the drive time
		sweeper = Sweeper(ge_rabi_params=None, ef_rabi_params=None, state_centers=None)
		return sweeper.sweep_parameter(
				sequence=rabi_sequence,
				sweep_range=self.time_range,
				frequencies=self.frequencies,
				shots=self.shots,
				interval=self.interval,
				plot=self.plot,
			)

	def analyze(self, system:System, result:ExperimentResult):
        # sweep data with the target labels
		sweep_data = result.data

        # fit the Rabi oscillation
		rabi_params = {
            target: fitting.fit_rabi(
                target=data.target,
                times=data.sweep_range,
                data=data.data,
                plot=self.plot,
            )
            for target, data in sweep_data.items()
        }

        # create the Rabi data for each target
		rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=self.time_range,
                rabi_param=rabi_params[target],
                state_centers=system.state_centers().get(target),
            )
            for target, data in sweep_data.items()
        }

        # create the experiment result
		result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )
