from __future__ import annotations

from collections import defaultdict
from typing import Final, Literal, Optional, Sequence

import numpy as np
from IPython.display import clear_output
from numpy.typing import NDArray
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from .. import fitting as fit
from .. import visualization as viz
from ..config import Config, Params, Qubit, Resonator, Target
from ..fitting import RabiParam
from ..hardware import Box
from ..measurement import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONTROL_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    Measurement,
    MeasureResult,
)
from ..pulse import Rect, Waveform
from ..typing import IQArray, ParametricWaveform, TargetMap
from .experiment_result import (
    AmplRabiRelation,
    ExperimentResult,
    FreqRabiRelation,
    SweepResult,
    TimePhaseRelation,
)
from .experiment_tool import ExperimentTool

console = Console()

MIN_DURATION = 128


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
        control_window: int = DEFAULT_CONTROL_WINDOW,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._control_window: Final = control_window
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
        self._rabi_params: Optional[dict[str, RabiParam]] = None

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Get the Rabi parameters."""
        if self._rabi_params is None:
            console.print("Rabi parameters are not stored.")
            return {}
        return self._rabi_params

    def store_rabi_params(self, rabi_params: dict[str, RabiParam]):
        """
        Stores the Rabi parameters.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        if self._rabi_params is not None:
            overwrite = Confirm.ask("Overwrite the existing Rabi parameters?")
            if not overwrite:
                return
        self._rabi_params = rabi_params
        console.print("Rabi parameters are stored.")

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

    def linkup(self) -> None:
        """Links up the measurement system."""
        box_list = list(self.boxes.keys())
        self._measurement.linkup(box_list)

    def measure(
        self,
        sequence: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given sequence.

        Parameters
        ----------
        sequence : TargetMap[IQArray]
            Sequence of the experiment.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
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
            mode=mode,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )
        if plot:
            for target, data in result.data.items():
                viz.plot_waveform(
                    data.raw,
                    sampling_period=8,  # TODO: set dynamically
                    title=f"Raw signal of {target}",
                    xlabel="Capture time (ns)",
                    ylabel="Amplitude (arb. unit)",
                )
        return result

    def _measure_batch(
        self,
        sequences: Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
    ):
        """
        Measures the signals using the given sequences.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray]]
            Sequences of the experiment.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
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
            mode=mode,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )

    def check_noise(
        self,
        targets: list[str],
        duration: int = 10240,
    ) -> MeasureResult:
        """
        Checks the noise level of the system.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the noise.
        duration : int, optional
            Duration of the noise measurement. Defaults to 2048.

        Returns
        -------
        MeasureResult
            Result of the experiment.
        """
        result = self._measurement.measure_noise(targets, duration)
        for target, data in result.data.items():
            viz.plot_waveform(
                np.array(data.raw, dtype=np.complex64) * 2 ** (-32),
                title=f"Readout noise of {target}",
                xlabel="Capture time (μs)",
                sampling_period=8e-3,
            )
        return result

    def check_waveform(
        self,
        targets: list[str],
    ) -> MeasureResult:
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the waveforms.

        Returns
        -------
        MeasureResult
            Result of the experiment.
        """
        result = self.measure(sequence={target: np.array([]) for target in targets})
        for target, data in result.data.items():
            viz.plot_waveform(
                data.raw,
                title=f"Readout waveform of {target}",
                sampling_period=8,
            )
        return result

    def check_rabi(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 201, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[SweepResult]:
        """
        Conducts a Rabi experiment with the default amplitude.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        time_range : NDArray, optional
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[SweepResult]
            Result of the experiment.
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=True,
        )
        return result

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: NDArray,
        detuning: float = 0.0,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[SweepResult]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        amplitudes : dict[str, float]
            Amplitudes of the control pulses.
        time_range : NDArray
            Time range of the experiment.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.0.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to False.

        Returns
        -------
        ExperimentResult[SweepResult]
            Result of the experiment.
        """
        targets = list(amplitudes.keys())
        time_range = np.array(time_range, dtype=np.int64)
        sequence = {
            target: lambda T: Rect(duration=T, amplitude=amplitudes[target]).detuned(
                detuning
            )
            for target in targets
        }
        result = self.sweep_parameter(
            sequence=sequence,
            sweep_range=time_range,
            sweep_value_label="Time (ns)",
            shots=shots,
            interval=interval,
            plot=plot,
        )
        rabi_params = self.fit_rabi(result.data)
        if store_params:
            self.store_rabi_params(rabi_params)
        result.rabi_params = rabi_params
        return result

    def sweep_parameter(
        self,
        *,
        sequence: TargetMap[ParametricWaveform],
        sweep_range: NDArray,
        sweep_value_label: str = "Sweep value",
        pulse_count=1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepResult]:
        """
        Sweeps a parameter and measures the signals.

        Parameters
        ----------
        sequence : TargetMap[ParametricWaveform]
            Parametric sequence to sweep.
        sweep_range : NDArray
            Range of the parameter to sweep.
        sweep_value_label : str
            Label of the sweep value.
        pulse_count : int, optional
            Number of pulses to apply. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[SweepResult]
            Result of the experiment.
        """
        targets = list(sequence.keys())
        sequences = [
            {
                target: sequence[target](param).repeated(pulse_count).values
                for target in targets
            }
            for param in sweep_range
        ]
        generator = self._measure_batch(
            sequences=sequences,
            shots=shots,
            interval=interval,
            control_window=self._control_window,
        )
        signals = defaultdict(list)
        for result in generator:
            for target, data in result.data.items():
                signals[target].append(data.kerneled)
            if plot:
                clear_output(wait=True)
                viz.scatter_iq_data(signals)
        data = {
            target: SweepResult(
                target=target,
                sweep_range=sweep_range,
                sweep_value_label=sweep_value_label,
                data=np.array(values),
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=data, rabi_params=self.rabi_params)
        return result

    def repeat_sequence(
        self,
        *,
        sequence: TargetMap[Waveform],
        n: int,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepResult]:
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
        ExperimentResult[SweepResult]
            Result of the experiment.
        """
        repeated_sequence = {
            target: lambda param, p=pulse: p.repeated(int(param))
            for target, pulse in sequence.items()
        }
        result = self.sweep_parameter(
            sweep_range=np.arange(n + 1),
            sweep_value_label="Number of repetitions",
            sequence=repeated_sequence,
            pulse_count=1,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        return result

    def obtain_freq_rabi_relation(
        self,
        targets: list[str],
        *,
        detuning_range: NDArray = np.linspace(-0.01, 0.01, 11),
        time_range: NDArray = np.arange(0, 101, 4),
    ) -> ExperimentResult[FreqRabiRelation]:
        """
        Obtains the relation between the detuning and the Rabi frequency.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        detuning_range : NDArray
            Range of the detuning to sweep in GHz.
        time_range : NDArray
            Time range of the experiment in ns.

        Returns
        -------
        ExperimentResult[FreqRabiRelation]
            Result of the experiment.
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for detuning in detuning_range:
            result = self.rabi_experiment(
                time_range=time_range,
                amplitudes=amplitudes,
                detuning=detuning,
                plot=False,
            )
            clear_output(wait=True)
            rabi_params = result.rabi_params
            if rabi_params is None:
                raise SystemError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)

        frequencies = {
            target: detuning_range + self.qubits[target].frequency for target in targets
        }

        data = {
            target: FreqRabiRelation(
                target=target,
                sweep_range=detuning_range,
                frequency_range=frequencies[target],
                data=np.array(values, dtype=np.float64),
            )
            for target, values in rabi_rates.items()
        }
        return ExperimentResult(data=data)

    def obtain_ampl_rabi_relation(
        self,
        targets: list[str],
        *,
        amplitude_range: NDArray = np.linspace(0.01, 0.1, 10),
        time_range: NDArray = np.arange(0, 201, 8),
    ) -> ExperimentResult[AmplRabiRelation]:
        """
        Obtains the relation between the control amplitude and the Rabi frequency.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        amplitude_range : NDArray
            Range of the control amplitude to sweep.

        Returns
        -------
        ExperimentResult[AmplRabiRelation]
            Result of the experiment.
        """

        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for amplitude in amplitude_range:
            if amplitude <= 0:
                continue
            result = self.rabi_experiment(
                time_range=time_range,
                amplitudes={target: amplitude for target in targets},
                plot=False,
            )
            clear_output(wait=True)
            rabi_params = result.rabi_params
            if rabi_params is None:
                raise SystemError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)

        data = {
            target: AmplRabiRelation(
                target=target,
                sweep_range=amplitude_range,
                data=np.array(values, dtype=np.float64),
            )
            for target, values in rabi_rates.items()
        }
        return ExperimentResult(data=data)

    def obtain_time_phase_relation(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 1024, 128),
    ) -> ExperimentResult[TimePhaseRelation]:
        """
        Obtains the relation between the control window and the phase shift.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the phase shift.
        time_range : NDArray, optional
            The control window range to sweep in ns.

        Returns
        -------
        ExperimentResult[PhaseShiftData]
            Result of the experiment.
        """
        results = defaultdict(list)
        for window in time_range:
            result = self.measure(
                sequence={target: np.zeros(0) for target in targets},
                control_window=window,
            )
            for qubit, value in result.data.items():
                iq = complex(value.kerneled)
                results[qubit].append(iq)
            clear_output(wait=True)
            viz.scatter_iq_data(results)
        data = {
            qubit: TimePhaseRelation(
                target=qubit,
                sweep_range=time_range,
                data=np.array(values),
            )
            for qubit, values in results.items()
        }
        return ExperimentResult(data=data)

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
        result: TargetMap[SweepResult],
        wave_count: Optional[float] = None,
        plot: bool = True,
    ) -> dict[str, RabiParam]:
        """
        Fits the measured data to a Rabi oscillation.

        Parameters
        ----------
        result : SweepResult
            Result of the Rabi experiment.
        wave_count : float, optional
            Number of waves in sweep_result. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

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
                plot=plot,
            )
            for target in result
        }
        return rabi_params

    def fit_damped_rabi(
        self,
        result: TargetMap[SweepResult],
        wave_count: Optional[float] = None,
        plot: bool = True,
    ) -> dict[str, RabiParam]:
        """
        Fits the measured data to a damped Rabi oscillation.

        Parameters
        ----------
        result : dict[str, SweepResult]
            Result of the Rabi experiment.
        wave_count : float, optional
            Number of waves in sweep_result. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

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
                plot=plot,
                is_damped=True,
            )
            for target in result
        }
        return rabi_params

    def calc_control_amplitudes(
        self,
        rabi_rate: float = 25e-3,
        rabi_params: dict[str, RabiParam] | None = None,
    ) -> dict[str, float]:
        """
        Calculates the control amplitudes for the Rabi rate.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam], optional
            Parameters of the Rabi oscillation. Defaults to None.
        rabi_rate : float, optional
            Rabi rate of the experiment. Defaults to 25 MHz.

        Returns
        -------
        dict[str, float]
            Control amplitudes for the Rabi rate.
        """
        current_amplitudes = self.params.control_amplitude
        rabi_params = rabi_params or self.rabi_params

        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / rabi_params[target].frequency
            for target in rabi_params
        }

        print(f"control_amplitude for {rabi_rate * 1e3} MHz\n")
        for target, amplitude in amplitudes.items():
            print(f"{target}: {amplitude:.6f}")

        print(f"\n{1/rabi_rate/4} ns rect pulse → π/2 pulse")

        return amplitudes