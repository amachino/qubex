from __future__ import annotations
from collections import defaultdict
from typing import Literal, Optional
import numpy as np
from numpy.typing import ArrayLike
from rich.console import Console
from ...analysis import IQPlotter
from ...measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from ...pulse import (
    PulseSchedule,

)
from ...typing import ParametricPulseSchedule, ParametricWaveformDict

from ...experiment.experiment_result import (
    ExperimentResult,
    SweepData,
)

console = Console()

USER_NOTE_PATH = ".user_note.json"
SYSTEM_NOTE_PATH = ".system_note.json"

STATE_CENTERS = "state_centers"
RABI_PARAMS = "rabi_params"
CR_PARAMS = "cr_params"

HPI_AMPLITUDE = "hpi_amplitude"
HPI_DURATION = 30
HPI_RAMPTIME = 10
PI_AMPLITUDE = "pi_amplitude"
PI_DURATION = 30
PI_RAMPTIME = 10
DRAG_HPI_AMPLITUDE = "drag_hpi_amplitude"
DRAG_HPI_BETA = "drag_hpi_beta"
DRAG_HPI_DURATION = 16
DRAG_PI_AMPLITUDE = "drag_pi_amplitude"
DRAG_PI_BETA = "drag_pi_beta"
DRAG_PI_DURATION = 16
DRAG_COEFF = 0.5

RABI_TIME_RANGE = range(0, 201, 4)
RABI_FREQUENCY = 0.0125
CALIBRATION_SHOTS = 2048

class Sweeper(object):
	def __init__(self, ge_rabi_params, ef_rabi_params, state_centers):
		self.ge_rabi_params = ge_rabi_params
		self.ef_rabi_params = ef_rabi_params
		self.state_centers = state_centers

	def sweep_parameter(
		self,
		sequence: ParametricPulseSchedule | ParametricWaveformDict,
		*,
		sweep_range: ArrayLike,
		repetitions: int = 1,
		frequencies: Optional[dict[str, float]] = None,
		rabi_level: Literal["ge", "ef"] = "ge",
		shots: int = DEFAULT_SHOTS,
		interval: int = DEFAULT_INTERVAL,
		control_window: int | None = None,
		capture_window: int | None = None,
		capture_margin: int | None = None,
		plot: bool = True,
		title: str = "Sweep result",
		xaxis_title: str = "Sweep value",
		yaxis_title: str = "Measured value",
		xaxis_type: Literal["linear", "log"] = "linear",
		yaxis_type: Literal["linear", "log"] = "linear",
	) -> ExperimentResult[SweepData]:
		"""
		Sweeps a parameter and measures the signals.

		Parameters
		----------
		sequence : ParametricPulseSchedule | ParametricWaveformMap
			Parametric sequence to sweep.
		sweep_range : ArrayLike
			Range of the parameter to sweep.
		repetitions : int, optional
			Number of repetitions. Defaults to 1.
		frequencies : Optional[dict[str, float]]
			Frequencies of the qubits.
		shots : int, optional
			Number of shots. Defaults to DEFAULT_SHOTS.
		interval : int, optional
			Interval between shots. Defaults to DEFAULT_INTERVAL.
		control_window : int, optional
			Control window. Defaults to None.
		capture_window : int, optional
			Capture window. Defaults to None.
		capture_margin : int, optional
			Capture margin. Defaults to None.
		plot : bool, optional
			Whether to plot the measured signals. Defaults to True.
		title : str, optional
			Title of the plot. Defaults to "Sweep result".
		xaxis_title : str, optional
			Title of the x-axis. Defaults to "Sweep value".
		yaxis_title : str, optional
			Title of the y-axis. Defaults to "Measured value".
		xaxis_type : Literal["linear", "log"], optional
			Type of the x-axis. Defaults to "linear".
		yaxis_type : Literal["linear", "log"], optional
			Type of the y-axis. Defaults to "linear".

		Returns
		-------
		ExperimentResult[SweepData]
			Result of the experiment.

		Examples
		--------
		>>> result = ex.sweep_parameter(
		...     sequence=lambda x: {"Q00": Rect(duration=30, amplitude=x)},
		...     sweep_range=np.arange(0, 101, 4),
		...     repetitions=4,
		...     shots=1024,
		...     plot=True,
		... )
		"""
		sweep_range = np.array(sweep_range)

		if rabi_level == "ge":
			rabi_params = self.ge_rabi_params
		elif rabi_level == "ef":
			rabi_params = self.ef_rabi_params
		else:
			raise ValueError("Invalid Rabi level.")

		if isinstance(sequence, dict):
			# TODO: this parameter type (dict[str, Callable[..., Waveform]]) will be deprecated
			targets = list(sequence.keys())
			sequences = [
				{
					target: sequence[target](param).repeated(repetitions).values
					for target in targets
				}
				for param in sweep_range
			]

		if callable(sequence):
			if isinstance(sequence(0), PulseSchedule):
				sequences = [
					sequence(param).repeated(repetitions).get_sampled_sequences()  # type: ignore
					for param in sweep_range
				]
			elif isinstance(sequence(0), dict):
				sequences = [
					{
						target: waveform.repeated(repetitions).values
						for target, waveform in sequence(param).items()  # type: ignore
					}
					for param in sweep_range
				]
		else:
			raise ValueError("Invalid sequence.")

		signals = defaultdict(list)
		plotter = IQPlotter(self.state_centers)

		generator = self._measure_batch(
			sequences=sequences,
			shots=shots,
			interval=interval,
			control_window=control_window or self._control_window,
			capture_window=capture_window or self._capture_window,
			capture_margin=capture_margin or self._capture_margin,
		)
		with self.modified_frequencies(frequencies):
			for result in generator:
				for target, data in result.data.items():
					signals[target].append(data.kerneled)
				if plot:
					plotter.update(signals)

		if plot:
			plotter.show()

		sweep_data = {
			target: SweepData(
				target=target,
				data=np.array(values),
				sweep_range=sweep_range,
				rabi_param=rabi_params.get(target),
				state_centers=self.state_centers.get(target),
				title=title,
				xaxis_title=xaxis_title,
				yaxis_title=yaxis_title,
				xaxis_type=xaxis_type,
				yaxis_type=yaxis_type,
			)
			for target, values in signals.items()
		}
		result = ExperimentResult(data=sweep_data, rabi_params=self.rabi_params)
		return result
