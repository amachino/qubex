from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Final, Literal, Optional, Sequence

import numpy as np
from IPython.display import clear_output
from numpy.typing import NDArray
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from .. import fitting as fit
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
from ..pulse import FlatTop, Rect, Waveform
from ..typing import IQArray, ParametricWaveform, TargetMap
from ..version import get_version
from ..visualization import IQPlotter, plot_waveform
from .experiment_record import ExperimentRecord
from .experiment_result import (
    AmplCalibData,
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    SweepData,
    TimePhaseData,
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
    control_window : int, optional
        Control window. Defaults to DEFAULT_CONTROL_WINDOW.
    config_dir : str, optional
        Directory of the configuration files. Defaults to DEFAULT_CONFIG_DIR.

    Examples
    --------
    >>> from qubex import Experiment
    >>> experiment = Experiment(
    ...     chip_id="64Q",
    ...     qubits=["Q00", "Q01"],
    ... )
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
        self._rabi_params: Optional[dict[str, RabiParam]] = None
        self._config: Final = Config(config_dir)
        self._measurement: Final = Measurement(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        self.tool: Final = ExperimentTool(
            chip_id=self._chip_id,
            qubits=self._qubits,
            config=self._config,
            measurement=self._measurement,
        )
        self.print_environment()

    @property
    def system(self):
        return self._config.get_quantum_system(self._chip_id)

    @property
    def params(self) -> Params:
        """Get the system parameters."""
        return self._config.get_params(self._chip_id)

    @property
    def chip_id(self) -> str:
        return self._chip_id

    @property
    def qubits(self) -> dict[str, Qubit]:
        all_qubits = self._config.get_qubits(self._chip_id)
        qubits = {}
        for qubit in all_qubits:
            if qubit.label in self._qubits:
                qubits[qubit.label] = qubit
        return qubits

    @property
    def resonators(self) -> dict[str, Resonator]:
        all_resonators = self._config.get_resonators(self._chip_id)
        resonators = {}
        for resonator in all_resonators:
            if resonator.qubit in self._qubits:
                resonators[resonator.qubit] = resonator
        return resonators

    @property
    def targets(self) -> dict[str, Target]:
        """Get the targets."""
        all_targets = self._measurement.targets
        targets = {}
        for target in all_targets:
            if all_targets[target].qubit in self._qubits:
                targets[target] = all_targets[target]
        return targets

    @property
    def boxes(self) -> dict[str, Box]:
        boxes = self._config.get_boxes_by_qubits(self._chip_id, self._qubits)
        return {box.id: box for box in boxes}

    @property
    def box_list(self) -> list[str]:
        return list(self.boxes.keys())

    @property
    def hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the default π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            π/2 pulse.
        """
        return {
            target: FlatTop(
                duration=30,
                amplitude=self.params.control_amplitude[target],
                tau=10,
            )
            for target in self.qubits
        }

    @property
    def pi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the default π pulse.

        Returns
        -------
        TargetMap[Waveform]
            π pulse.
        """
        return {
            target: FlatTop(
                duration=30,
                amplitude=self.params.control_amplitude[target],
                tau=10,
            ).repeated(2)
            for target in self.qubits
        }

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Get the Rabi parameters."""
        if self._rabi_params is None:
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

    def print_environment(self):
        print("date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("python:", sys.version.split()[0])
        print("env:", sys.prefix)
        print("qubex:", get_version())
        print("config:", self._config.config_path)
        print("chip:", self._chip_id)
        print("qubits:", ", ".join(self.qubits))
        print("boxes:", ", ".join(self.boxes))
        print("control_window:", self._control_window, "ns")

    def print_resources(self):
        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in self.boxes.values():
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    def check_status(self):
        link_status = self._measurement.check_link_status(self.box_list)
        clock_status = self._measurement.check_clock_status(self.box_list)
        if link_status["status"]:
            console.print("Link status: OK", style="green")
        else:
            console.print("Link status: NG", style="red")
        console.print(link_status["links"])
        if clock_status["status"]:
            console.print("Clock status: OK", style="green")
        else:
            console.print("Clock status: NG", style="red")
        console.print(clock_status["clocks"])

    def linkup(
        self,
        box_list: Optional[list[str]] = None,
    ) -> None:
        """
        Link up the measurement system.

        Parameters
        ----------
        box_list : Optional[list[str]], optional
            List of the box IDs to link up. Defaults to None.

        Examples
        --------
        >>> experiment.linkup()
        """
        if box_list is None:
            box_list = self.box_list
        self._measurement.linkup(box_list)

    def relinkup(
        self,
        box_list: Optional[list[str]] = None,
    ) -> None:
        """
        Relink up the measurement system.

        Parameters
        ----------
        box_list : Optional[list[str]], optional
            List of the box IDs to link up. Defaults to None.

        Examples
        --------
        >>> experiment.relinkup()
        """
        if box_list is None:
            box_list = self.box_list
        self._measurement.relinkup(box_list)
        self.check_status()

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]):
        """
        Temporarily modifies the frequencies of the qubits.

        Parameters
        ----------
        frequencies : dict[str, float]
            Modified frequencies in GHz.

        Examples
        --------
        >>> with ex.modified_frequencies({"Q00": 5.0}):
        ...     # Do something
        """
        with self._measurement.modified_frequencies(frequencies):
            yield

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord:
        """
        Load an experiment record from a file.

        Parameters
        ----------
        name : str
            Name of the experiment record to load.

        Returns
        -------
        ExperimentRecord
            The loaded ExperimentRecord instance.

        Raises
        ------
        FileNotFoundError

        Examples
        --------
        >>> record = experiment.load_record("some_record.json")
        """
        record = ExperimentRecord.load(name)
        print(f"ExperimentRecord `{name}` is loaded.\n")
        print(f"description: {record.description}")
        print(f"created_at: {record.created_at}")
        return record

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

        Examples
        --------
        >>> result = experiment.measure(
        ...     sequence={"Q00": np.zeros(0)},
        ...     mode="avg",
        ...     shots=3000,
        ...     interval=100 * 1024,
        ...     control_window=1024,
        ...     plot=True,
        ... )
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
            result.plot()
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
        *,
        duration: int = 10240,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the noise level of the system.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the noise.
        duration : int, optional
            Duration of the noise measurement. Defaults to 2048.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.check_noise(["Q00", "Q01"])
        """
        result = self._measurement.measure_noise(targets, duration)
        for target, data in result.data.items():
            if plot:
                plot_waveform(
                    np.array(data.raw, dtype=np.complex64) * 2 ** (-32),
                    title=f"Readout noise of {target}",
                    xlabel="Capture time (μs)",
                    sampling_period=8e-3,
                )
        return result

    def check_waveform(
        self,
        targets: list[str],
        *,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the waveforms.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.check_waveform(["Q00", "Q01"])
        """
        result = self.measure(sequence={target: np.zeros(0) for target in targets})
        if plot:
            result.plot()
        return result

    def check_rabi(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 201, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
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
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.check_rabi(["Q00", "Q01"])
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=True,
            plot=plot,
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
    ) -> ExperimentResult[RabiData]:
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
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.rabi_experiment(
        ...     amplitudes={"Q00": 0.1},
        ...     time_range=np.arange(0, 201, 4),
        ...     detuning=0.0,
        ...     shots=1024,
        ... )
        """
        targets = list(amplitudes.keys())
        time_range = np.array(time_range, dtype=np.float64)

        def rabi_sequence(target: str) -> ParametricWaveform:
            return lambda T: Rect(
                duration=T,
                amplitude=amplitudes[target],
            )

        sequence = {target: rabi_sequence(target) for target in targets}

        detuned_frequencies = {
            target: self.targets[target].frequency + detuning for target in amplitudes
        }
        with self.modified_frequencies(detuned_frequencies):
            sweep_result = self.sweep_parameter(
                sequence=sequence,
                sweep_range=time_range,
                sweep_value_label="Time (ns)",
                shots=shots,
                interval=interval,
                plot=plot,
            )
        rabi_params = {
            target: fit.fit_rabi(
                target=data.target,
                times=data.sweep_range,
                data=data.data,
                plot=plot,
            )
            for target, data in sweep_result.data.items()
        }
        if store_params:
            self.store_rabi_params(rabi_params)
        rabi_data = {
            target: RabiData(
                target=target,
                data=sweep_result.data[target].data,
                time_range=time_range,
                rabi_param=rabi_params[target],
            )
            for target in targets
        }
        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )
        return result

    def sweep_parameter(
        self,
        *,
        sequence: TargetMap[ParametricWaveform],
        sweep_range: NDArray,
        sweep_value_label: str = "Sweep value",
        repetitions: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
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
        repetitions : int, optional
            Number of repetitions. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.sweep_parameter(
        ...     sequence={"Q00": lambda x: Rect(duration=30, amplitude=x)},
        ...     sweep_range=np.arange(0, 101, 4),
        ...     repetitions=4,
        ...     shots=1024,
        ...     plot=True,
        ... )
        """
        targets = list(sequence.keys())
        sequences = [
            {
                target: sequence[target](param).repeated(repetitions).values
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
        plotter = IQPlotter()
        for result in generator:
            for target, data in result.data.items():
                signals[target].append(data.kerneled)
            if plot:
                plotter.update(signals)
        data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                sweep_value_label=sweep_value_label,
                rabi_param=self.rabi_params.get(target),
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=data, rabi_params=self.rabi_params)
        return result

    def repeat_sequence(
        self,
        *,
        sequence: TargetMap[Waveform],
        repetitions: int,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
        """
        Repeats the pulse sequence n times.

        Parameters
        ----------
        sequence : dict[str, Waveform]
            Pulse sequence to repeat.
        repetitions : int
            Number of repetitions.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.repeat_sequence(
        ...     sequence={"Q00": Rect(duration=64, amplitude=0.1)},
        ...     repetitions=4,
        ... )
        """
        repeated_sequence = {
            target: lambda param, p=pulse: p.repeated(int(param))
            for target, pulse in sequence.items()
        }
        result = self.sweep_parameter(
            sweep_range=np.arange(repetitions + 1),
            sweep_value_label="Number of repetitions",
            sequence=repeated_sequence,
            repetitions=1,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        return result

    def obtain_freq_rabi_relation(
        self,
        targets: list[str],
        *,
        detuning_range: NDArray = np.linspace(-0.01, 0.01, 15),
        time_range: NDArray = np.arange(0, 101, 4),
        plot: bool = True,
    ) -> ExperimentResult[FreqRabiData]:
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
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[FreqRabiData]
            Result of the experiment.

        Raises
        ------
        ValueError
            If the Rabi parameters are not stored.

        Examples
        --------
        >>> result = experiment.obtain_freq_rabi_relation(
        ...     targets=["Q00", "Q01"],
        ...     detuning_range=np.linspace(-0.01, 0.01, 11),
        ...     time_range=np.arange(0, 101, 4),
        ... )
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for detuning in detuning_range:
            result = self.rabi_experiment(
                time_range=time_range,
                amplitudes=amplitudes,
                detuning=detuning,
                plot=plot,
            )
            clear_output(wait=True)
            rabi_params = result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)

        frequencies = {
            target: detuning_range + self.qubits[target].frequency for target in targets
        }

        data = {
            target: FreqRabiData(
                target=target,
                data=np.array(values, dtype=np.float64),
                sweep_range=detuning_range,
                frequency_range=frequencies[target],
            )
            for target, values in rabi_rates.items()
        }
        return ExperimentResult(data=data)

    def obtain_ampl_rabi_relation(
        self,
        targets: list[str],
        *,
        amplitude_range: NDArray = np.linspace(0.01, 0.1, 10),
        time_range: NDArray = np.arange(0, 201, 4),
        plot: bool = True,
    ) -> ExperimentResult[AmplRabiData]:
        """
        Obtains the relation between the control amplitude and the Rabi frequency.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        amplitude_range : NDArray
            Range of the control amplitude to sweep.
        time_range : NDArray
            Time range of the experiment in ns.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[AmplRabiData]
            Result of the experiment.

        Raises
        ------
        ValueError
            If the Rabi parameters are not stored.

        Examples
        --------
        >>> result = experiment.obtain_ampl_rabi_relation(
        ...     targets=["Q00", "Q01"],
        ...     amplitude_range=np.linspace(0.01, 0.1, 10),
        ...     time_range=np.arange(0, 201, 4),
        ... )
        """

        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for amplitude in amplitude_range:
            if amplitude <= 0:
                continue
            result = self.rabi_experiment(
                time_range=time_range,
                amplitudes={target: amplitude for target in targets},
                plot=plot,
            )
            clear_output(wait=True)
            rabi_params = result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)
        data = {
            target: AmplRabiData(
                target=target,
                data=np.array(values, dtype=np.float64),
                sweep_range=amplitude_range,
            )
            for target, values in rabi_rates.items()
        }
        return ExperimentResult(data=data)

    def obtain_time_phase_relation(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 1024, 128),
        plot: bool = True,
    ) -> ExperimentResult[TimePhaseData]:
        """
        Obtains the relation between the control window and the phase shift.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the phase shift.
        time_range : NDArray, optional
            The control window range to sweep in ns.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[PhaseShiftData]
            Result of the experiment.

        Examples
        --------
        >>> result = experiment.obtain_time_phase_relation(
        ...     targets=["Q00", "Q01"],
        ...     time_range=np.arange(0, 1024, 128),
        ... )
        """
        iq_data = defaultdict(list)
        plotter = IQPlotter()
        for window in time_range:
            result = self.measure(
                sequence={target: np.zeros(0) for target in targets},
                control_window=window,
                plot=False,
            )
            for qubit, value in result.data.items():
                iq = complex(value.kerneled)
                iq_data[qubit].append(iq)
            if plot:
                plotter.update(iq_data)
        data = {
            qubit: TimePhaseData(
                target=qubit,
                data=np.array(values),
                sweep_range=time_range,
            )
            for qubit, values in iq_data.items()
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

    def calc_control_amplitudes(
        self,
        rabi_rate: float = 12.5e-3,
        rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool = True,
    ) -> dict[str, float]:
        """
        Calculates the control amplitudes for the Rabi rate.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam], optional
            Parameters of the Rabi oscillation. Defaults to None.
        rabi_rate : float, optional
            Rabi rate of the experiment. Defaults to 12.5 MHz.
        print_result : bool, optional
            Whether to print the result. Defaults to True.

        Returns
        -------
        dict[str, float]
            Control amplitudes for the Rabi rate.
        """
        current_amplitudes = self.params.control_amplitude
        rabi_params = rabi_params or self.rabi_params

        if self._rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / rabi_params[target].frequency
            for target in rabi_params
        }

        if print_result:
            print(f"control_amplitude for {rabi_rate * 1e3} MHz\n")
            for target, amplitude in amplitudes.items():
                print(f"{target}: {amplitude:.6f}")

            print(f"\n{1/rabi_rate/4} ns rect pulse → π/2 pulse")

        return amplitudes

    def calibrate_hpi_pulse(
        self,
        targets: list[str],
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate the π/2 pulse.

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.
        """
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            rabi_rate = 12.5e-3
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]
            ampl_min = ampl * 0.5
            ampl_max = ampl * 1.5
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            result = self.sweep_parameter(
                sequence={
                    target: lambda x: FlatTop(
                        duration=30,
                        amplitude=x,
                        tau=10,
                    )
                },
                sweep_range=ampl_range,
                sweep_value_label="Control amplitude",
                repetitions=4,
                shots=DEFAULT_SHOTS,
                interval=DEFAULT_INTERVAL,
            ).data[target]
            return AmplCalibData(
                target=target,
                data=result.normalized,
                sweep_range=result.sweep_range,
            )

        data = {}
        for target in targets:
            data[target] = calibrate(target)
            clear_output(wait=True)

        return ExperimentResult(data=data)
