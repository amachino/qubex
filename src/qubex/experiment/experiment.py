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

from ..analysis import (
    IQPlotter,
    RabiParam,
    display_bloch_sphere,
    fitting,
    plot_state_vectors,
    plot_waveform,
)
from ..clifford import CliffordGroup
from ..config import Config, Params, Qubit, Resonator, Target
from ..measurement import (
    DEFAULT_CAPTURE_WINDOW,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONTROL_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
    DEFAULT_SHOTS,
    Measurement,
    MeasureResult,
    StateClassifier,
)
from ..pulse import (
    CPMG,
    Blank,
    Drag,
    FlatTop,
    PhaseShift,
    Pulse,
    PulseSequence,
    Rect,
    VirtualZ,
    Waveform,
)
from ..typing import IQArray, ParametricWaveform, TargetMap
from ..version import get_package_version
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import (
    AmplCalibData,
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    RamseyData,
    RBData,
    SweepData,
    T1Data,
    T2Data,
    TimePhaseData,
)
from .experiment_tool import ExperimentTool

console = Console()

MIN_DURATION = 128

USER_NOTE_PATH = ".user_note.json"
SYSTEM_NOTE_PATH = ".system_note.json"

DEFAULT_HPI_AMPLITUDE = "default_hpi_amplitude"
DEFAULT_HPI_DURATION = 30
DEFAULT_HPI_RISETIME = 10
DEFAULT_PI_AMPLITUDE = "default_pi_amplitude"
DEFAULT_PI_DURATION = 30
DEFAULT_PI_RISETIME = 10
DRAG_HPI_AMPLITUDE = "drag_hpi_amplitude"
DRAG_HPI_DURATION = 16
DRAG_HPI_LAMBDA = 0.5
DRAG_PI_AMPLITUDE = "drag_pi_amplitude"
DRAG_PI_DURATION = 16
DRAG_PI_LAMBDA = 0.5


class Experiment:
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    qubits : list[str]
        List of qubits to use in the experiment.
    config_dir : str, optional
        Directory of the configuration files. Defaults to DEFAULT_CONFIG_DIR.
    control_window : int, optional
        Control window. Defaults to DEFAULT_CONTROL_WINDOW.
    capture_window : int, optional
        Capture window. Defaults to DEFAULT_CAPTURE_WINDOW.
    readout_duration : int, optional
        Readout duration. Defaults to DEFAULT_READOUT_DURATION.

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
        config_dir: str = DEFAULT_CONFIG_DIR,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ):
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._control_window: Final = control_window
        self._capture_window: Final = capture_window
        self._readout_duration: Final = readout_duration
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
        self._user_note: Final = ExperimentNote(
            file_path=USER_NOTE_PATH,
        )
        self._system_note: Final = ExperimentNote(
            file_path=SYSTEM_NOTE_PATH,
        )
        self.print_environment()

    @property
    def system(self):
        """Get the quantum system."""
        return self._config.get_quantum_system(self._chip_id)

    @property
    def params(self) -> Params:
        """Get the system parameters."""
        return self._config.get_params(self._chip_id)

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self._chip_id

    @property
    def qubits(self) -> dict[str, Qubit]:
        """Get the qubits."""
        all_qubits = self._config.get_qubits(self._chip_id)
        qubits = {}
        for qubit in all_qubits:
            if qubit.label in self._qubits:
                qubits[qubit.label] = qubit
        return qubits

    @property
    def resonators(self) -> dict[str, Resonator]:
        """Get the resonators."""
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
    def box_list(self) -> list[str]:
        """Get the list of the box IDs."""
        boxes = self._config.get_boxes_by_qubits(self._chip_id, self._qubits)
        return [box.id for box in boxes]

    @property
    def config_path(self) -> str:
        """Get the path of the configuration file."""
        return str(self._config.config_path)

    @property
    def note(self) -> ExperimentNote:
        """Get the user note."""
        return self._user_note

    @property
    def hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the default π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            π/2 pulse.
        """
        # preset hpi amplitude
        amplitude = self.params.control_amplitude
        # calibrated hpi amplitude
        calib_amplitude: dict[str, float] = self._system_note.get(DEFAULT_HPI_AMPLITUDE)
        if calib_amplitude is not None:
            for target in calib_amplitude:
                # use the calibrated hpi amplitude if it is stored
                amplitude[target] = calib_amplitude[target]
        return {
            target: FlatTop(
                duration=DEFAULT_HPI_DURATION,
                amplitude=amplitude[target],
                tau=DEFAULT_HPI_RISETIME,
            )
            for target in self._qubits
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
        # preset hpi pulse
        hpi = self.hpi_pulse
        # generate the pi pulse from the hpi pulse
        pi = {target: hpi[target].repeated(2) for target in self._qubits}
        # calibrated pi amplitude
        calib_amplitude: dict[str, float] = self._system_note.get(DEFAULT_PI_AMPLITUDE)
        if calib_amplitude is not None:
            for target in calib_amplitude:
                # use the calibrated pi amplitude if it is stored
                pi[target] = FlatTop(
                    duration=DEFAULT_PI_DURATION,
                    amplitude=calib_amplitude[target],
                    tau=DEFAULT_PI_RISETIME,
                )
        return {target: pi[target] for target in self._qubits}

    @property
    def drag_hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the DRAG π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            DRAG π/2 pulse.
        """
        calib_amplitude: dict[str, float] = self._system_note.get(DRAG_HPI_AMPLITUDE)
        if calib_amplitude is None:
            raise ValueError("DRAG HPI amplitude is not stored.")
        return {
            target: Drag(
                duration=DRAG_HPI_DURATION,
                amplitude=calib_amplitude[target],
                beta=-DRAG_HPI_LAMBDA / self.qubits[target].anharmonicity,
            )
            for target in self._qubits
        }

    @property
    def drag_pi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the DRAG π pulse.

        Returns
        -------
        TargetMap[Waveform]
            DRAG π pulse.
        """
        calib_amplitude: dict[str, float] = self._system_note.get(DRAG_PI_AMPLITUDE)
        if calib_amplitude is None:
            raise ValueError("DRAG PI amplitude is not stored.")
        return {
            target: Drag(
                duration=DRAG_PI_DURATION,
                amplitude=calib_amplitude[target],
                beta=-DRAG_PI_LAMBDA / self.qubits[target].anharmonicity,
            )
            for target in self._qubits
        }

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Get the Rabi parameters."""
        if self._rabi_params is None:
            return {}
        return self._rabi_params

    def _validate_rabi_params(self):
        """Check if the Rabi parameters are stored."""
        if self._rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

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

    def get_pulse_for_state(
        self,
        target: str,
        state: Literal["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform:
        """
        Get the pulse to prepare the given state from the ground state.

        Parameters
        ----------
        target : str
            Target qubit.
        state : Literal["0", "1", "+", "-", "+i", "-i"]
            State to prepare.

        Returns
        -------
        Waveform
            Pulse for the state.
        """
        if state == "0":
            return Blank(0)
        elif state == "1":
            return self.pi_pulse[target]
        else:
            hpi = self.hpi_pulse[target]
            if state == "+":
                return hpi.shifted(np.pi / 2)
            elif state == "-":
                return hpi.shifted(-np.pi / 2)
            elif state == "+i":
                return hpi.shifted(np.pi)
            elif state == "-i":
                return hpi
            else:
                raise ValueError("Invalid state.")

    def get_spectators(self, qubit: str) -> list[Qubit]:
        """
        Get the spectators of the given qubit.

        Parameters
        ----------
        qubit : str
            Qubit to get the spectators.

        Returns
        -------
        list[Qubit]
            List of the spectators.
        """
        spectator_labels = self.system.chip.graph.get_spectators(qubit)
        spectators: list[Qubit] = []
        for label in spectator_labels:
            spectator = self._config.get_qubit(self.chip_id, label)
            spectators.append(spectator)
        return spectators

    def print_environment(self, verbose: bool = False):
        """Print the environment information."""
        print("date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("python:", sys.version.split()[0])
        print("env:", sys.prefix)
        if verbose:
            print("numpy:", get_package_version("numpy"))
            print("quel_ic_config:", get_package_version("quel_ic_config"))
            print("quel_clock_master:", get_package_version("quel_clock_master"))
            print("qubecalib:", get_package_version("qubecalib"))
        print("qubex:", get_package_version("qubex"))
        print("config:", self._config.config_path)
        print("chip:", self._chip_id)
        print("qubits:", self._qubits)
        print("control_window:", self._control_window, "ns")
        print("")
        print("Following devices will be used:")
        self.print_boxes()

    def print_boxes(self):
        """Print the box information."""
        boxes = self._config.get_boxes_by_qubits(self._chip_id, self._qubits)
        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in boxes:
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    def check_status(self):
        """Check the status of the measurement system."""
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
        >>> ex.linkup()
        """
        if box_list is None:
            box_list = self.box_list
        self._measurement.linkup(box_list)
        self.check_status()

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float] | None):
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
        if frequencies is None:
            yield
        else:
            with self._measurement.modified_frequencies(frequencies):
                yield

    def save_default(self):
        """Save the default settings."""
        self._system_note.save()

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
        >>> record = ex.load_record("some_record.json")
        """
        record = ExperimentRecord.load(name)
        print(f"ExperimentRecord `{name}` is loaded.\n")
        print(f"description: {record.description}")
        print(f"created_at: {record.created_at}")
        return record

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform],
        *,
        frequencies: Optional[dict[str, float]] = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        readout_duration: int | None = None,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given sequence.

        Parameters
        ----------
        sequence : TargetMap[IQArray]
            Sequence of the experiment.
        frequencies : Optional[dict[str, float]]
            Frequencies of the qubits.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.measure(
        ...     sequence={"Q00": np.zeros(0)},
        ...     mode="avg",
        ...     shots=3000,
        ...     interval=100 * 1024,
        ...     control_window=1024,
        ...     plot=True,
        ... )
        """
        control_window = control_window or self._control_window
        capture_window = capture_window or self._capture_window
        readout_duration = readout_duration or self._readout_duration
        waveforms = {}
        for target, waveform in sequence.items():
            if isinstance(waveform, Waveform):
                waveforms[target] = waveform.values
            else:
                waveforms[target] = np.array(waveform, dtype=np.complex128)

        if frequencies is None:
            result = self._measurement.measure(
                waveforms=waveforms,
                mode=mode,
                shots=shots,
                interval=interval,
                control_window=control_window,
                capture_window=capture_window,
                readout_duration=readout_duration,
            )
        else:
            with self.modified_frequencies(frequencies):
                result = self._measurement.measure(
                    waveforms=waveforms,
                    mode=mode,
                    shots=shots,
                    interval=interval,
                    control_window=control_window,
                    capture_window=capture_window,
                    readout_duration=readout_duration,
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
        control_window: int | None = None,
        capture_window: int | None = None,
        readout_duration: int | None = None,
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
            Control window. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.

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
            control_window=control_window or self._control_window,
            capture_window=capture_window or self._capture_window,
            readout_duration=readout_duration or self._readout_duration,
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
        >>> result = ex.check_noise(["Q00", "Q01"])
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
        >>> result = ex.check_waveform(["Q00", "Q01"])
        """
        result = self.measure(sequence={target: np.zeros(0) for target in targets})
        if plot:
            result.plot()
        return result

    def check_rabi(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 201, 8),
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
        >>> result = ex.check_rabi(["Q00", "Q01"])
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
        >>> result = ex.rabi_experiment(
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
        sweep_result = self.sweep_parameter(
            sequence=sequence,
            sweep_range=time_range,
            frequencies=detuned_frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        rabi_params = {
            target: fitting.fit_rabi(
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
        sequence: TargetMap[ParametricWaveform],
        *,
        sweep_range: NDArray,
        repetitions: int = 1,
        frequencies: Optional[dict[str, float]] = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
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
        sequence : TargetMap[ParametricWaveform]
            Parametric sequence to sweep.
        sweep_range : NDArray
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
            control_window=control_window or self._control_window,
        )
        signals = defaultdict(list)
        plotter = IQPlotter()
        with self.modified_frequencies(frequencies):
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
                rabi_param=self.rabi_params.get(target),
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=data, rabi_params=self.rabi_params)
        return result

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform],
        *,
        repetitions: int = 10,
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
        repetitions : int, optional
            Number of repetitions. Defaults to 10.
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
        >>> result = ex.repeat_sequence(
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
            sequence=repeated_sequence,
            repetitions=1,
            shots=shots,
            interval=interval,
            plot=plot,
            xaxis_title="Number of repetitions",
        )
        return result

    def obtain_freq_rabi_relation(
        self,
        targets: list[str],
        *,
        detuning_range: NDArray = np.linspace(-0.01, 0.01, 15),
        time_range: NDArray = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
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
        >>> result = ex.obtain_freq_rabi_relation(
        ...     targets=["Q00", "Q01"],
        ...     detuning_range=np.linspace(-0.01, 0.01, 11),
        ...     time_range=np.arange(0, 101, 4),
        ... )
        """
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for detuning in detuning_range:
            rabi_result = self.rabi_experiment(
                time_range=time_range,
                amplitudes=amplitudes,
                detuning=detuning,
                shots=shots,
                interval=interval,
                plot=False,
            )
            clear_output()
            if plot:
                rabi_result.fit()
            clear_output(wait=True)
            rabi_params = rabi_result.rabi_params
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

        result = ExperimentResult(data=data)
        return result

    def obtain_ampl_rabi_relation(
        self,
        targets: list[str],
        *,
        amplitude_range: NDArray = np.linspace(0.01, 0.1, 10),
        time_range: NDArray = np.arange(0, 201, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
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
        >>> result = ex.obtain_ampl_rabi_relation(
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
                amplitudes={target: amplitude for target in targets},
                time_range=time_range,
                shots=shots,
                interval=interval,
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
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[PhaseShiftData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.obtain_time_phase_relation(
        ...     targets=["Q00", "Q01"],
        ...     time_range=np.arange(0, 1024, 128),
        ... )
        """
        iq_data = defaultdict(list)
        plotter = IQPlotter()
        for window in time_range:
            result = self.measure(
                sequence={target: np.zeros(0) for target in targets},
                shots=shots,
                interval=interval,
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

        self._validate_rabi_params()

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

    def calibrate_control_frequency(
        self,
        targets: list[str],
        *,
        detuning_range: NDArray = np.linspace(-0.01, 0.01, 15),
        time_range: NDArray = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        result = self.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        fit_data = {target: data.fit()[0] for target, data in result.data.items()}
        for target, fit in fit_data.items():
            print(f"{target}: {fit:.6f}")
        return fit_data

    def calibrate_readout_frequency(
        self,
        targets: list[str],
        *,
        detuning_range: NDArray = np.linspace(-0.01, 0.01, 15),
        time_range: NDArray = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, list[float]]:
        result = defaultdict(list)
        for detuning in detuning_range:
            modified_frequencies = {
                resonator.label: resonator.frequency + detuning
                for resonator in self.resonators.values()
            }
            with self.modified_frequencies(modified_frequencies):
                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes={
                        target: self.params.control_amplitude[target]
                        for target in targets
                    },
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                for qubit, data in rabi_result.data.items():
                    result[qubit].append(data.rabi_param.amplitude)
                clear_output(wait=True)
        return result

    def calibrate_default_pulse(
        self,
        targets: list[str],
        pulse_type: Literal["pi", "hpi"],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=DEFAULT_HPI_DURATION,
                    amplitude=1,
                    tau=DEFAULT_HPI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=DEFAULT_PI_DURATION,
                    amplitude=1,
                    tau=DEFAULT_PI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]
            ampl_min = ampl * 0.5
            ampl_max = ampl * 1.5
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            sweep_data = self.sweep_parameter(
                sequence={target: lambda x: pulse.scaled(x)},
                sweep_range=ampl_range,
                repetitions=2 if pulse_type == "pi" else 4,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            calib_value = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=-sweep_data.normalized,
                title=f"{pulse_type} pulse calibration",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=calib_value,
            )

        data: dict[str, AmplCalibData] = {}
        for idx, target in enumerate(targets):
            print(f"[{idx+1}/{len(targets)}] Calibrating {target}...\n")
            data[target] = calibrate(target)
            print("")

        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"{target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_drag_pulse(
        self,
        targets: list[str],
        pulse_type: Literal["pi", "hpi"],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = Drag(
                    duration=DRAG_HPI_DURATION,
                    amplitude=1,
                    beta=-DRAG_HPI_LAMBDA / self.qubits[target].anharmonicity,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = Drag(
                    duration=DRAG_PI_DURATION,
                    amplitude=1,
                    beta=-DRAG_PI_LAMBDA / self.qubits[target].anharmonicity,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]
            ampl_min = ampl * 0.5
            ampl_max = ampl * 1.25  # stop at 1.25 to avoid leakage
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            sweep_data = self.sweep_parameter(
                sequence={
                    target: lambda x: pulse.scaled(x),
                },
                sweep_range=ampl_range,
                repetitions=2 if pulse_type == "pi" else 4,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            calib_value = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=-sweep_data.normalized,
                title=f"DRAG {pulse_type} pulse calibration",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=calib_value,
            )

        data: dict[str, AmplCalibData] = {}
        for idx, target in enumerate(targets):
            print(f"[{idx+1}/{len(targets)}] Calibrating {target}...\n")
            data[target] = calibrate(target)
            print("")

        print(f"Calibration results for DRAG {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"{target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_hpi_pulse(
        self,
        targets: list[str],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="hpi",
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(DEFAULT_HPI_AMPLITUDE, ampl)

        return result

    def calibrate_pi_pulse(
        self,
        targets: list[str],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="pi",
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(DEFAULT_PI_AMPLITUDE, ampl)

        return result

    def calibrate_drag_hpi_pulse(
        self,
        targets: list[str],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_drag_pulse(
            targets=targets,
            pulse_type="hpi",
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(DRAG_HPI_AMPLITUDE, ampl)

        return result

    def calibrate_drag_pi_pulse(
        self,
        targets: list[str],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG π pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_drag_pulse(
            targets=targets,
            pulse_type="pi",
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(DRAG_PI_AMPLITUDE, ampl)

        return result

    def t1_experiment(
        self,
        targets: list[str],
        *,
        time_range: NDArray = 2 ** np.arange(1, 18),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[T1Data]:
        """
        Conducts a T1 experiment in parallel.

        Parameters
        ----------
        targets : list[str]
            List of qubits to check the T1 decay.
        time_range : NDArray
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[T1Data]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.t1_experiment(
        ...     target="Q00",
        ...     time_range=2 ** np.arange(1, 18),
        ...     shots=1024,
        ... )
        """

        # wrap the lambda function with a function to scope the qubit variable
        def t1_sequence(target: str) -> ParametricWaveform:
            return lambda T: PulseSequence(
                [
                    self.pi_pulse[target],
                    Blank(T),
                ]
            )

        t1_sequences = {target: t1_sequence(target) for target in targets}

        sweep_result = self.sweep_parameter(
            sequence=t1_sequences,
            sweep_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
            title="T1 decay",
            xaxis_title="Time (μs)",
            yaxis_title="Measured value",
            xaxis_type="log",
        )

        t1_value = {
            target: fitting.fit_exp_decay(
                target=target,
                x=data.sweep_range,
                y=0.5 * (1 - data.normalized),
                title="T1",
                xaxis_title="Time (μs)",
                yaxis_title="Population",
                xaxis_type="log",
                yaxis_type="linear",
            )
            for target, data in sweep_result.data.items()
        }

        data = {
            target: T1Data.new(data, t1_value[target])
            for target, data in sweep_result.data.items()
        }

        return ExperimentResult(data=data)

    def t2_experiment(
        self,
        targets: list[str],
        *,
        time_range: NDArray = 200 * 2 ** np.arange(10),
        n_cpmg: int = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[T2Data]:
        """
        Conducts a T2 experiment in series.

        Parameters
        ----------
        qubits : list[str]
            List of qubits to check the T2 decay.
        time_range : NDArray
            Time range of the experiment in ns.
        n_cpmg : int, optional
            Number of CPMG pulses. Defaults to 1.
        pi_cpmg : Waveform, optional
            π pulse for the CPMG sequence. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[T2Data]
            Result of the experiment.
        """

        # wrap the lambda function with a function to scope the qubit variable
        def t2_sequence(target: str) -> ParametricWaveform:
            hpi = self.hpi_pulse[target]
            pi = pi_cpmg or hpi.repeated(2)

            def waveform(T: int) -> Waveform:
                if T == 0:
                    return PulseSequence(
                        [
                            hpi,
                            hpi.shifted(np.pi),
                        ]
                    )
                return PulseSequence(
                    [
                        hpi,
                        # Blank((T - pi.duration) // 2),
                        # pi,
                        # Blank((T - pi.duration) // 2),
                        CPMG(
                            tau=(T - pi.duration * n_cpmg) // (2 * n_cpmg),
                            pi=pi,
                            n=n_cpmg,
                        ),
                        hpi.shifted(np.pi),
                    ]
                )

            return waveform

        data: dict[str, T2Data] = {}
        for target in targets:
            sweep_data = self.sweep_parameter(
                sequence={target: t2_sequence(target)},
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
            ).data[target]
            t2 = fitting.fit_exp_decay(
                target=target,
                x=sweep_data.sweep_range,
                y=0.5 * (1 - sweep_data.normalized),
                title="T2",
                xaxis_title="Time (μs)",
                yaxis_title="Population",
            )
            t2_data = T2Data.new(sweep_data, t2=t2)
            data[target] = t2_data

        return ExperimentResult(data=data)

    def ramsey_experiment(
        self,
        targets: list[str],
        *,
        time_range: NDArray = np.arange(0, 10000, 200),
        detuning: float = 0.0005,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RamseyData]:
        """
        Conducts a Ramsey experiment in series.

        Parameters
        ----------
        qubits : list[str]
            List of qubits to check the Ramsey oscillation.
        time_range : NDArray
            Time range of the experiment in ns.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.001 GHz.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RamseyData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.ramsey_experiment(
        ...     target="Q00",
        ...     time_range=np.arange(0, 10000, 100),
        ...     shots=1024,
        ... )
        """

        def ramsey_sequence(target: str) -> dict[str, ParametricWaveform]:
            hpi = self.hpi_pulse[target]
            sequence: dict[str, ParametricWaveform] = {
                target: lambda T: PulseSequence(
                    [
                        hpi,
                        Blank(T),
                        hpi.shifted(np.pi),
                    ]
                )
            }
            if spectator_state != "0":
                spectators = self.get_spectators(target)
                for spectator in spectators:
                    if spectator.label in self._qubits:
                        pulse = self.get_pulse_for_state(
                            target=spectator.label,
                            state=spectator_state,
                        )
                        sequence[spectator.label] = lambda T: PulseSequence(
                            [
                                pulse,
                                Blank(sequence[target](T).duration),
                            ]
                        )
            return sequence

        data: dict[str, RamseyData] = {}
        for target in targets:
            detuned_frequency = self.qubits[target].frequency + detuning

            sweep_data = self.sweep_parameter(
                sequence=ramsey_sequence(target),
                sweep_range=time_range,
                frequencies={target: detuned_frequency},
                shots=shots,
                interval=interval,
                plot=plot,
            ).data[target]
            t2, ramsey_freq = fitting.fit_ramsey(
                target=target,
                x=sweep_data.sweep_range,
                y=sweep_data.normalized,
            )
            ramsey_data = RamseyData.new(
                sweep_data=sweep_data,
                t2=t2,
                ramsey_freq=ramsey_freq,
            )
            data[target] = ramsey_data

        return ExperimentResult(data=data)

    def build_classifier(
        self,
        targets: list[str],
    ):
        result_g = self.measure(
            {target: np.zeros(0) for target in targets},
            mode="single",
        )
        result_e = self.measure(
            {target: self.pi_pulse[target].values for target in targets},
            mode="single",
        )
        self._measurement.classifiers = {
            target: StateClassifier.fit(
                {
                    0: result_g.data[target].kerneled,
                    1: result_e.data[target].kerneled,
                }
            )
            for target in targets
        }
        for target in targets:
            clf = self._measurement.classifiers[target]
            clf.classify(result_g.data[target].kerneled)
            clf.classify(result_e.data[target].kerneled)

    def rb_sequence(
        self,
        *,
        target: str,
        n: int,
        x90: Waveform | None = None,
        interleave_waveform: Waveform | None = None,
        interleave_map: dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseSequence:
        """
        Generates a randomized benchmarking sequence.

        Parameters
        ----------
        target : str
            Target qubit.
        n : int
            Number of Clifford gates.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        interleave_waveform : Waveform, optional
            Waveform of the interleaved gate. Defaults to None.
        interleave_map : dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        seed : int, optional
            Random seed.

        Returns
        -------
        PulseSequence
            Randomized benchmarking sequence.

        Examples
        --------
        >>> sequence = ex.rb_sequence(
        ...     target="Q00",
        ...     n=100,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ... )

        >>> sequence = ex.rb_sequence(
        ...     target="Q00",
        ...     n=100,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     interleave_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleave_map={
        ...         "I": (1, "I"),
        ...         "X": (1, "X"),
        ...         "Y": (-1, "Y"),
        ...         "Z": (-1, "Z"),
        ...     },
        ... )
        """
        x90 = x90 or self.hpi_pulse[target]
        z90 = VirtualZ(np.pi / 2)

        sequence: list[Waveform | VirtualZ] = []

        clifford_group = CliffordGroup()

        if interleave_waveform is None:
            cliffords, inverse = clifford_group.create_rb_sequences(
                n=n,
                seed=seed,
            )
        else:
            if interleave_map is None:
                raise ValueError("Interleave map must be provided.")
            cliffords, inverse = clifford_group.create_irb_sequences(
                n=n,
                seed=seed,
                interleave=interleave_map,
            )

        for clifford in cliffords:
            for gate in clifford:
                if gate == "X90":
                    sequence.append(x90)
                elif gate == "Z90":
                    sequence.append(z90)
            if interleave_waveform is not None:
                sequence.append(interleave_waveform)

        for gate in inverse:
            if gate == "X90":
                sequence.append(x90)
            elif gate == "Z90":
                sequence.append(z90)
        return PulseSequence(sequence)

    def rb_experiment(
        self,
        *,
        target: str,
        n_cliffords_range: NDArray[np.int64] = np.arange(0, 1001, 50),
        x90: Waveform | None = None,
        interleave_waveform: Waveform | None = None,
        interleave_map: dict[str, tuple[complex, str]] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seed: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RBData]:
        """
        Conducts a randomized benchmarking experiment.

        Parameters
        ----------
        target : str
            Target qubit.
        n_cliffords_range : NDArray[np.int64], optional
            Range of the number of Cliffords. Defaults to np.arange(0, 1001, 50).
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        interleave_waveform : Waveform, optional
            Waveform of the interleaved gate. Defaults to None.
        interleave_map : dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seed : int, optional
            Random seed.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RBData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.rb_experiment(
        ...     target="Q00",
        ...     n_cliffords_range=np.arange(0, 1001, 50),
        ...     x90=Rect(duration=30, amplitude=0.1),
        ... )

        >>> result = ex.rb_experiment(
        ...     target="Q00",
        ...     n_cliffords_range=np.arange(0, 1001, 50),
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     interleave_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleave_map={
        ...         "I": (1, "I"),
        ...         "X": (1, "X"),
        ...         "Y": (-1, "Y"),
        ...         "Z": (-1, "Z"),
        ...     },
        ... )
        """

        def rb_sequence(target: str) -> dict[str, ParametricWaveform]:
            sequence: dict[str, ParametricWaveform] = {
                target: lambda N: self.rb_sequence(
                    target=target,
                    n=N,
                    x90=x90,
                    interleave_waveform=interleave_waveform,
                    interleave_map=interleave_map,
                    seed=seed,
                )
            }
            if spectator_state != "0":
                spectators = self.get_spectators(target)
                for spectator in spectators:
                    if spectator.label in self._qubits:
                        pulse = self.get_pulse_for_state(
                            target=spectator.label,
                            state=spectator_state,
                        )
                        sequence[spectator.label] = lambda N: PulseSequence(
                            [
                                pulse,
                                Blank(sequence[target](N).duration),
                            ]
                        )
            return sequence

        sweep_result = self.sweep_parameter(
            rb_sequence(target),
            sweep_range=n_cliffords_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        sweep_data = sweep_result.data[target]

        fit_data = fitting.fit_rb(
            target=target,
            x=sweep_data.sweep_range,
            y=sweep_data.normalized,
            title="Randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Z expectation value",
            xaxis_type="linear",
            yaxis_type="linear",
        )

        data = {
            qubit: RBData.new(
                data,
                depolarizing_rate=fit_data[0],
                avg_gate_error=fit_data[1],
                avg_gate_fidelity=fit_data[2],
            )
            for qubit, data in sweep_result.data.items()
        }

        return ExperimentResult(data=data)

    def randomized_benchmarking(
        self,
        target: str,
        *,
        n_cliffords_range: NDArray[np.int64] = np.arange(0, 1001, 100),
        n_trials: int = 30,
        x90: Waveform | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        n_cliffords_range : NDArray[np.int64], optional
            Range of the number of Cliffords. Defaults to np.arange(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seed : int, optional
            Random seed.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment.
        """
        self._validate_rabi_params()

        results = []
        seeds = np.random.randint(0, 2**32, n_trials)
        for seed in seeds:
            result = self.rb_experiment(
                target=target,
                n_cliffords_range=n_cliffords_range,
                spectator_state=spectator_state,
                x90=x90,
                seed=seed,
                shots=shots,
                interval=interval,
                plot=False,
            )
            results.append(result.data[target].normalized)
            clear_output(wait=True)

        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)

        fit_result = fitting.fit_rb(
            target=target,
            x=n_cliffords_range,
            y=mean,
            error_y=std,
            plot=plot,
            title="Randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Z expectation value",
            xaxis_type="linear",
            yaxis_type="linear",
        )

        return {
            "depolarizing_rate": fit_result[0],
            "avg_gate_error": fit_result[1],
            "avg_gate_fidelity": fit_result[2],
            "n_cliffords": n_cliffords_range,
            "mean": mean,
            "std": std,
        }

    def interleaved_randomized_benchmarking(
        self,
        *,
        target: str,
        interleave_waveform: Waveform,
        interleave_map: dict[str, tuple[complex, str]],
        n_cliffords_range: NDArray[np.int64] = np.arange(0, 1001, 100),
        n_trials: int = 30,
        x90: Waveform | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        interleave_waveform : Waveform
            Waveform of the interleaved gate.
        interleave_map : dict[str, tuple[complex, str]]
            Clifford map of the interleaved gate.
        n_cliffords_range : NDArray[np.int64], optional
            Range of the number of Cliffords. Defaults to np.arange(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seed : int, optional
            Random seed.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment.
        """
        results = []
        seeds = np.random.randint(0, 2**32, n_trials)
        for seed in seeds:
            result = self.rb_experiment(
                target=target,
                n_cliffords_range=n_cliffords_range,
                x90=x90,
                interleave_waveform=interleave_waveform,
                interleave_map=interleave_map,
                spectator_state=spectator_state,
                seed=seed,
                shots=shots,
                interval=interval,
                plot=False,
            )
            results.append(result.data[target].normalized)
            clear_output(wait=True)

        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)

        fit_result = fitting.fit_rb(
            target=target,
            x=n_cliffords_range,
            y=mean,
            error_y=std,
            plot=plot,
            title="Interleaved randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Z expectation value",
            xaxis_type="linear",
            yaxis_type="linear",
        )

        return {
            "depolarizing_rate": fit_result[0],
            "avg_gate_error": fit_result[1],
            "avg_gate_fidelity": fit_result[2],
            "n_cliffords": n_cliffords_range,
            "mean": mean,
            "std": std,
        }

    def state_tomography_sequence(
        self,
        *,
        target: str,
        sequence: IQArray | Waveform,
        basis: str,
        x90: Waveform | None = None,
    ) -> PulseSequence:
        """
        Generates a state tomography sequence.

        Parameters
        ----------
        target : str
            Target qubit.
        sequence : IQArray | Waveform
            Sequence to measure.
        basis : str
            Measurement basis. "X", "Y", or "Z".
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.

        Returns
        -------
        PulseSequence
            State tomography sequence.
        """
        if isinstance(sequence, list) or isinstance(sequence, np.ndarray):
            sequence = Pulse(sequence)
        elif not isinstance(sequence, Waveform):
            raise ValueError("Invalid sequence.")

        x90 = x90 or self.hpi_pulse[target]
        y90m = x90.shifted(-np.pi / 2)

        if basis == "X":
            if isinstance(sequence, PulseSequence):
                return sequence.added(y90m)
            else:
                return PulseSequence([sequence, y90m])
        elif basis == "Y":
            if isinstance(sequence, PulseSequence):
                return sequence.added(x90)
            else:
                return PulseSequence([sequence, x90])
        elif basis == "Z":
            return PulseSequence([sequence])
        else:
            raise ValueError("Invalid basis.")

    def state_tomography(
        self,
        sequence: TargetMap[IQArray | Waveform],
        *,
        x90: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Conducts a state tomography experiment.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform]
            Sequence to measure for each target.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Results of the experiment.
        """
        buffer: dict[str, list[float]] = defaultdict(list)
        for basis in ["X", "Y", "Z"]:
            measure_result = self.measure(
                {
                    target: self.state_tomography_sequence(
                        target=target,
                        sequence=sequence,
                        basis=basis,
                        x90=x90,
                    )
                    for target, sequence in sequence.items()
                },
                shots=shots,
                interval=interval,
                plot=plot,
            )
            for target, data in measure_result.data.items():
                rabi_param = self.rabi_params[target]
                if rabi_param is None:
                    raise ValueError("Rabi parameters are not stored.")
                values = data.kerneled
                values_rotated = values * np.exp(-1j * rabi_param.angle)
                values_normalized = (
                    np.imag(values_rotated) - rabi_param.offset
                ) / rabi_param.amplitude
                buffer[target] += [values_normalized]

        result = {
            target: (
                values[0],  # X
                values[1],  # Y
                values[2],  # Z
            )
            for target, values in buffer.items()
        }
        return result

    def state_evolution_tomography(
        self,
        *,
        sequences: Sequence[TargetMap[IQArray | Waveform]],
        x90: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Conducts a state evolution tomography experiment.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray | Waveform]]
            Sequences to measure for each target.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Results of the experiment.
        """
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for sequence in sequences:
            state_vectors = self.state_tomography(
                sequence=sequence,
                x90=x90,
                shots=shots,
                interval=interval,
                plot=False,
            )
            for target, state_vector in state_vectors.items():
                buffer[target].append(state_vector)

        result = {target: np.array(states) for target, states in buffer.items()}

        if plot:
            for target, states in result.items():
                print(f"State evolution of {target}")
                display_bloch_sphere(states)

        return result

    def pulse_tomography(
        self,
        waveforms: TargetMap[IQArray | Waveform],
        *,
        x90: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> TargetMap[NDArray[np.float64]]:
        """
        Conducts a pulse tomography experiment.

        Parameters
        ----------
        waveforms : TargetMap[IQArray | Waveform]
            Waveforms to measure for each target.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        """
        self._validate_rabi_params()

        pulses: dict[str, Waveform] = {}
        pulse_length_set = set()
        for target, waveform in waveforms.items():
            if isinstance(waveform, Waveform):
                pulse = waveform
            elif isinstance(waveform, list) or isinstance(waveform, np.ndarray):
                pulse = Pulse(waveform)
            else:
                raise ValueError("Invalid waveform.")
            pulses[target] = pulse
            pulse_length_set.add(pulse.length)
        if len(pulse_length_set) != 1:
            raise ValueError("The lengths of the waveforms must be the same.")

        pulse_length = pulse_length_set.pop()

        if plot:
            for target in pulses:
                pulses[target].plot(title=f"Waveform of {target}")

        def partial_waveform(waveform: Waveform, index: int) -> Waveform:
            """Returns a partial waveform up to the given index."""

            # If the waveform is a PulseSequence, we need to handle the PhaseShift gate.
            if isinstance(waveform, PulseSequence):
                current_index = 0
                pulse_sequence = PulseSequence([])
                for pulse in waveform._sequence:
                    # If the pulse is a PhaseShift gate, we can simply add it to the sequence.
                    if isinstance(pulse, PhaseShift):
                        pulse_sequence = pulse_sequence.added(pulse)
                        continue
                    # If the pulse is a Pulse and the length is greater than the index, we need to create a partial pulse.
                    elif current_index + pulse.length > index:
                        pulse = Pulse(pulse.values[0 : index - current_index])
                        pulse_sequence = pulse_sequence.added(pulse)
                        break
                    # If the pulse is a Pulse and the length is less than the index, we can add the pulse to the sequence.
                    else:
                        pulse_sequence = pulse_sequence.added(pulse)
                        current_index += pulse.length
                return pulse_sequence
            # If the waveform is a Pulse, we can simply return the partial waveform.
            else:
                return Pulse(waveform.values[0:index])

        sequences = [
            {target: partial_waveform(pulse, i) for target, pulse in pulses.items()}
            for i in range(pulse_length + 1)
        ]

        result = self.state_evolution_tomography(
            sequences=sequences,
            x90=x90,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        if plot:
            for target, states in result.items():
                plot_state_vectors(
                    times=pulses.popitem()[1].times,
                    state_vectors=states,
                    title=f"State evolution of {target}",
                )

        return result
