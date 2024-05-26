from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Final, Optional

import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from . import fitting as fit
from . import visualization as viz
from .config import Config, Params, Qubit, Resonator, Target
from .experiment_tool import ExperimentTool
from .fitting import RabiParam
from .hardware import Box
from .measurement import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONTROL_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    Measurement,
    MeasureResult,
)
from .pulse import Rect, Waveform

console = Console()

MIN_DURATION = 128


@dataclass
class SweepResult:
    """
    Data class representing the result of a sweep experiment.

    Attributes
    ----------
    target : str
        Target of the experiment.
    sweep_range : NDArray
        Sweep range of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    target: str
    sweep_range: NDArray
    data: NDArray
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rotated(self, param: RabiParam) -> NDArray:
        return self.data * np.exp(-1j * param.angle)

    def normalized(self, param: RabiParam) -> NDArray:
        values = self.data * np.exp(-1j * param.angle)
        values_normalized = (values.imag - param.offset) / param.amplitude
        return values_normalized

    def plot(self, rabi_params: RabiParam):
        values = self.normalized(rabi_params)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.sweep_range,
                y=values,
                mode="lines+markers",
                marker=dict(symbol="circle", size=8, color="#636EFA"),
                line=dict(width=1, color="grey", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Rabi oscillation of {self.target}",
            xaxis_title="Sweep value",
            yaxis_title="Normalized value",
            width=600,
        )
        fig.show()


class Experiment:
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    qubits : list[str]
        List of qubits to use in the experiment.
    data_dir : str, optional
        Path to the directory where the experiment data is stored. Defaults to "./data".
    """

    def __init__(
        self,
        *,
        chip_id: str,
        qubits: list[str],
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._config: Final = Config(config_dir)
        self._measurement: Final = Measurement(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        self.tool: Final = ExperimentTool(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        self.system: Final = self._config.get_quantum_system(chip_id)
        self.print_resources()

    def print_resources(self):
        console.print("The following resources will be used:\n")
        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in self.boxes.values():
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self._chip_id

    @property
    def qubits(self) -> dict[str, Qubit]:
        all_qubits = self._config.get_qubits(self._chip_id)
        return {
            qubit.label: qubit for qubit in all_qubits if qubit.label in self._qubits
        }

    @property
    def params(self) -> Params:
        """Get the system parameters."""
        return self._config.get_params(self._chip_id)

    @property
    def resonators(self) -> dict[str, Resonator]:
        all_resonators = self._config.get_resonators(self._chip_id)
        return {
            resonator.qubit: resonator
            for resonator in all_resonators
            if resonator.qubit in self._qubits
        }

    @property
    def targets(self) -> dict[str, Target]:
        all_targets = self._config.get_all_targets(self._chip_id)
        targets = [target for target in all_targets if target.qubit in self._qubits]
        return {target.label: target for target in targets}

    @property
    def boxes(self) -> dict[str, Box]:
        boxes = self._config.get_boxes_by_qubits(self._chip_id, self._qubits)
        return {box.id: box for box in boxes}

    def connect(self) -> None:
        """Connect to the backend."""
        box_list = list(self.boxes.keys())
        self._measurement.connect(box_list)

    def measure(
        self,
        sequence: dict[str, NDArray[np.complex128]],
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given sequence.

        Parameters
        ----------
        sequence : dict[str, NDArray[np.complex128]]
            Sequence of the experiment.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to DEFAULT_CONTROL_WINDOW.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.
        """
        waveforms = {
            target: np.array(waveform, dtype=np.complex128)
            for target, waveform in sequence.items()
        }
        result = self._measurement.measure(
            waveforms=waveforms,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )
        if plot:
            for target, data in result.raw.items():
                viz.plot_waveform(
                    data,
                    sampling_period=8,  # TODO: set dynamically
                    title=f"Raw signal of {target}",
                    xlabel="Capture time (ns)",
                    ylabel="Amplitude (arb. unit)",
                )
        return result

    def _measure_batch(
        self,
        sequences: list[dict[str, NDArray[np.complex128]]],
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
    ):
        """
        Measures the signals using the given sequences.

        Parameters
        ----------
        sequences : list[dict[str, NDArray[np.complex128]]]
            List of sequences to measure.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to DEFAULT_CONTROL_WINDOW.

        Yields
        ------
        MeasureResult
            Result of the experiment.
        """
        waveforms_list = [
            {
                target: np.array(waveform, dtype=np.complex128)
                for target, waveform in sequence.items()
            }
            for sequence in sequences
        ]
        return self._measurement.measure_batch(
            waveforms_list=waveforms_list,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )

    def check_noise(
        self,
        targets: list[str],
        duration: int = 2048,
    ):
        """
        Checks the noise level of the system.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the noise.
        duration : int, optional
            Duration of the noise measurement. Defaults to 2048.
        """
        result = self._measurement.measure_noise(targets, duration)
        for target, data in result.raw.items():
            viz.plot_waveform(
                np.array(data, dtype=np.complex64) * 2 ** (-32),
                title=f"Readout noise of {target}",
                sampling_period=8,
            )

    def check_waveform(
        self,
        targets: list[str],
    ):
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the waveforms.
        """
        result = self.measure(sequence={target: np.array([]) for target in targets})
        for target, data in result.raw.items():
            viz.plot_waveform(
                data,
                title=f"Readout waveform of {target}",
                sampling_period=8,
            )

    def check_rabi(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, RabiParam]:
        """
        Conducts a Rabi experiment with the default amplitude.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        time_range : NDArray, optional
            Time range of the experiment. Defaults to np.arange(0, 201, 10).
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
        )

        rabi_params = self.fit_rabi(result)

        return rabi_params

    def rabi_experiment(
        self,
        *,
        time_range: NDArray,
        amplitudes: dict[str, float],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        time_range : NDArray
            Time range of the experiment.
        amplitudes : dict[str, float]
            Amplitudes of the control pulses.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        targets = list(amplitudes.keys())
        time_range = np.array(time_range, dtype=np.int64)
        control_window = MIN_DURATION * (max(time_range) // MIN_DURATION + 1)
        waveforms_list = [
            {
                target: Rect(
                    duration=T,
                    amplitude=amplitudes[target],
                ).values
                for target in targets
            }
            for T in time_range
        ]
        generator = self._measurement.measure_batch(
            waveforms_list=waveforms_list,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )

        signals = defaultdict(list)
        for result in generator:
            for target, data in result.kerneled.items():
                signals[target].append(data)
            if plot:
                clear_output(wait=True)
                viz.scatter_iq_data(signals)
        results = {
            target: SweepResult(
                target=target,
                sweep_range=time_range,
                data=np.array(values),
            )
            for target, values in signals.items()
        }
        return results

    def sweep_parameter(
        self,
        *,
        param_range: NDArray,
        sequence: dict[str, Callable[..., Waveform]],
        pulse_count=1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Sweeps a parameter and measures the signals.

        Parameters
        ----------
        param_range : NDArray
            Range of the parameter to sweep.
        sequence : dict[str, Callable[..., Waveform]]
            Parametric sequence to sweep.
        pulse_count : int, optional
            Number of pulses to apply. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to DEFAULT_CONTROL_WINDOW.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        targets = list(sequence.keys())
        sequences = [
            {
                target: sequence[target](param).repeated(pulse_count).values
                for target in targets
            }
            for param in param_range
        ]
        generator = self._measure_batch(
            sequences=sequences,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )
        signals = defaultdict(list)
        for result in generator:
            for target, data in result.kerneled.items():
                signals[target].append(data)
            if plot:
                viz.scatter_iq_data(signals)
        results = {
            target: SweepResult(
                target=target,
                sweep_range=param_range,
                data=np.array(values),
            )
            for target, values in signals.items()
        }
        return results

    def repeat_sequence(
        self,
        *,
        sequence: dict[str, Waveform],
        n: int,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Repeats the pulse sequence n times.

        Parameters
        ----------
        sequence : dict[str, Waveform]
            Pulse sequence to repeat.
        n : int
            Number of times to repeat the pulse.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        repeated_sequence = {
            target: lambda param, p=pulse: p.repeated(int(param))
            for target, pulse in sequence.items()
        }
        result = self.sweep_parameter(
            param_range=np.arange(n + 1),
            sequence=repeated_sequence,
            pulse_count=1,
            shots=shots,
            interval=interval,
        )
        return result

    def normalize(
        self,
        value: complex,
        param: RabiParam,
    ) -> float:
        """
        Normalizes the measured I/Q value.

        Parameters
        ----------
        value : complex
            Measured I/Q value.
        param : RabiParam
            Parameters of the Rabi oscillation.

        Returns
        -------
        float
            Normalized value.
        """
        value_rotated = value * np.exp(-1j * param.angle)
        value_normalized = (value_rotated.imag - param.offset) / param.amplitude
        return value_normalized

    def fit_rabi(
        self,
        result: dict[str, SweepResult],
        wave_count: Optional[float] = None,
    ) -> dict[str, RabiParam]:
        """
        Fits the measured data to a Rabi oscillation.

        Parameters
        ----------
        result : SweepResult
            Result of the Rabi experiment.
        wave_count : float, optional
            Number of waves in sweep_result. Defaults to None.

        Returns
        -------
        dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        rabi_params = {
            target: fit.fit_rabi(
                target=result[target].target,
                times=result[target].sweep_range,
                data=result[target].data,
                wave_count=wave_count,
            )
            for target in result
        }
        return rabi_params

    def fit_damped_rabi(
        self,
        result: dict[str, SweepResult],
        wave_count: Optional[float] = None,
    ) -> dict[str, RabiParam]:
        """
        Fits the measured data to a damped Rabi oscillation.

        Parameters
        ----------
        result : dict[str, SweepResult]
            Result of the Rabi experiment.
        wave_count : float, optional
            Number of waves in sweep_result. Defaults to None.

        Returns
        -------
        dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        rabi_params = {
            target: fit.fit_rabi(
                target=result[target].target,
                times=result[target].sweep_range,
                data=result[target].data,
                wave_count=wave_count,
                is_damped=True,
            )
            for target in result
        }
        return rabi_params

    def calc_control_amplitudes(
        self,
        rabi_params: dict[str, RabiParam],
        rabi_rate: float = 25e-3,
    ) -> dict[str, float]:
        """
        Calculates the control amplitudes for the Rabi rate.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        rabi_rate : float, optional
            Rabi rate of the experiment. Defaults to 25 MHz.

        Returns
        -------
        dict[str, float]
            Control amplitudes for the Rabi rate.
        """
        current_amplitudes = self.params.control_amplitude
        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / rabi_params[target].frequency
            for target in rabi_params
        }

        print(f"control_amplitude for {rabi_rate * 1e3} MHz\n")
        for target, amplitude in amplitudes.items():
            print(f"{target}: {amplitude:.6f}")

        print(f"\n{1/rabi_rate/4} ns rect pulse will be π/2 pulse")

        return amplitudes
