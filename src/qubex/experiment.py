from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Final, Optional

import numpy as np
from numpy.typing import NDArray
from qubecalib import QubeCalib
from rich.console import Console
from rich.table import Table

from .analysis import fit_damped_rabi, fit_rabi
from .config import Config, Params, Qubit, Resonator
from .hardware import Box, Port
from .measurement import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONTROL_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    MeasResult,
    Measurement,
)
from .pulse import Rect, Waveform
from .visualization import scatter_iq_data

console = Console()

DEFAULT_DATA_DIR = "data"
MIN_DURATION = 128


@dataclass
class RabiParams:
    """
    Data class representing the parameters of Rabi oscillation.

    Attributes
    ----------
    qubit : str
        Identifier of the qubit.
    phase_shift : float
        Phase shift of the I/Q signal.
    fluctuation : float
        Fluctuation of the I/Q signal.
    amplitude : float
        Amplitude of the Rabi oscillation.
    omega : float
        Angular frequency of the Rabi oscillation.
    phi : float
        Phase of the Rabi oscillation.
    offset : float
        Offset of the Rabi oscillation.
    """

    qubit: str
    phase_shift: float
    fluctuation: float
    amplitude: float
    omega: float
    phi: float
    offset: float


@dataclass
class SweepResult:
    """
    Data class representing the result of a sweep experiment.

    Attributes
    ----------
    qubit : str
        Identifier of the qubit.
    sweep_range : NDArray
        Sweep range of the experiment.
    data : NDArray
        Measured data.
    created_at : str
        Time when the experiment is conducted.
    """

    qubit: str
    sweep_range: NDArray
    data: NDArray
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rotated(self, rabi_params: RabiParams) -> NDArray:
        return self.data * np.exp(-1j * rabi_params.phase_shift)

    def normalized(self, rabi_params: RabiParams) -> NDArray:
        values = self.data * np.exp(-1j * rabi_params.phase_shift)
        values_normalized = -(values.imag - rabi_params.offset) / rabi_params.amplitude
        return values_normalized


class Experiment:
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    data_dir : str, optional
        Path to the directory where the experiment data is stored. Defaults to "./data".
    """

    def __init__(
        self,
        *,
        chip_id: str,
        data_dir: str = DEFAULT_DATA_DIR,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        self._chip_id: Final = chip_id
        self._config: Final = Config(config_dir)
        self._data_dir: Final = data_dir
        self._measurement: Final = Measurement(
            chip_id,
            config_dir=config_dir,
        )
        self.tool = ExperimentTool(self)

    @property
    def available_boxes(self) -> list[str]:
        """Get the list of available boxes."""
        return [box.id for box in self.boxes]

    @property
    def available_qubits(self) -> list[str]:
        """Get the list of available qubits."""
        return [qubit.label for qubit in self.qubits]

    @property
    def targets(self) -> dict[str, float]:
        """Get the list of target frequencies."""
        return self._measurement.targets

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self._chip_id

    @property
    def qubits(self) -> list[Qubit]:
        """Get the list of qubits."""
        return self._config.get_qubits(self._chip_id)

    @property
    def resonators(self) -> list[Resonator]:
        """Get the list of resonators."""
        return self._config.get_resonators(self._chip_id)

    @property
    def params(self) -> Params:
        """Get the system parameters."""
        return self._config.get_params(self._chip_id)

    @property
    def boxes(self) -> list[Box]:
        """Get the list of boxes."""
        return self._config.get_boxes(self._chip_id)

    @property
    def ports(self) -> list[Port]:
        """Get the list of ports."""
        return self._config.get_port_details(self._chip_id)

    def connect(self) -> None:
        """Connect to the backend."""
        self._measurement.connect()

    def measure(
        self,
        sequence: dict[str, NDArray[np.complex128]],
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
    ) -> MeasResult:
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

        Returns
        -------
        MeasResult
            Result of the experiment.
        """
        waveforms = {
            qubit: np.array(waveform, dtype=np.complex128)
            for qubit, waveform in sequence.items()
        }
        result = self._measurement.measure(
            waveforms=waveforms,
            shots=shots,
            interval=interval,
            control_window=control_window,
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
        MeasResult
            Result of the experiment.
        """
        waveforms_list = [
            {
                qubit: np.array(waveform, dtype=np.complex128)
                for qubit, waveform in sequence.items()
            }
            for sequence in sequences
        ]
        return self._measurement.measure_batch(
            waveforms_list=waveforms_list,
            shots=shots,
            interval=interval,
            control_window=control_window,
        )

    def rabi_experiment(
        self,
        *,
        time_range: list[int] | NDArray[np.int64],
        amplitudes: dict[str, float],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        time_range : list[int] | NDArray[np.int64]
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
        qubits = list(amplitudes.keys())
        time_range = np.array(time_range, dtype=np.int64)
        control_window = MIN_DURATION * (max(time_range) // MIN_DURATION + 1)
        waveforms_list = [
            {
                qubit: Rect(
                    duration=T,
                    amplitude=amplitudes[qubit],
                ).values
                for qubit in qubits
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
            for qubit, data in result.kerneled.items():
                signals[qubit].append(data)
            if plot:
                scatter_iq_data(signals)
        results = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=time_range,
                data=np.array(values),
            )
            for qubit, values in signals.items()
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
        qubits = list(sequence.keys())
        sequences = [
            {
                qubit: sequence[qubit](param).repeated(pulse_count).values
                for qubit in qubits
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
            for qubit, data in result.kerneled.items():
                signals[qubit].append(data)
            if plot:
                scatter_iq_data(signals)
        results = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=param_range,
                data=np.array(values),
            )
            for qubit, values in signals.items()
        }
        return results

    def rabi_check(
        self,
        qubits: list[str],
        *,
        time_range=np.arange(0, 201, 10),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, SweepResult]:
        """
        Conducts a Rabi experiment with the default amplitude.

        Parameters
        ----------
        quibits : list[str]
            List of qubits to check the Rabi oscillation.
        time_range : NDArray, optional
            Time range of the experiment. Defaults to np.arange(0, 201, 10).
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        ampl = self.params.control_amplitude
        amplitudes = {qubit: ampl[qubit] for qubit in qubits}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
        )
        return result

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
            qubit: lambda param, p=pulse: p.repeated(int(param))
            for qubit, pulse in sequence.items()
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
        iq_value: complex,
        rabi_params: RabiParams,
    ) -> float:
        """
        Normalizes the measured IQ value.

        Parameters
        ----------
        iq_value : complex
            Measured IQ value.
        rabi_params : RabiParams
            Parameters of the Rabi oscillation.

        Returns
        -------
        float
            Normalized value.
        """
        iq_value = iq_value * np.exp(-1j * rabi_params.phase_shift)
        value = iq_value.imag
        value = -(value - rabi_params.offset) / rabi_params.amplitude
        return value

    def fit_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Fits the measured data to a Rabi oscillation.

        Parameters
        ----------
        data : SweepResult
            Measured data.
        wave_count : float, optional
            Number of waves to fit. Defaults to 2.5.

        Returns
        -------
        RabiParams
            Parameters of the Rabi oscillation.
        """
        times = data.sweep_range
        signals = data.data

        phase_shift, fluctuation, popt = fit_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[1],
            phi=popt[2],
            offset=popt[3],
        )
        return rabi_params

    def fit_damped_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Fits the measured data to a damped Rabi oscillation.

        Parameters
        ----------
        data : SweepResult
            Measured data.
        wave_count : float, optional
            Number of waves to fit. Defaults to 2.5.

        Returns
        -------
        RabiParams
            Parameters of the Rabi oscillation.
        """
        times = data.sweep_range
        signals = data.data

        phase_shift, fluctuation, popt = fit_damped_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[2],
            phi=popt[3],
            offset=popt[4],
        )
        return rabi_params


class ExperimentTool:

    def __init__(self, experiment: Experiment):
        self._exp = experiment

    def get_qubecalib(self) -> QubeCalib:
        """Get the QubeCalib instance."""
        return self._exp._measurement._backend.qubecalib

    def dump_box(self, box_id: str) -> dict:
        """Dump the information of a box."""
        return self._exp._measurement.dump_box_config(box_id)

    def configure_boxes(self, box_list: Optional[list[str]] = None) -> None:
        """
        Configure the boxes.

        Parameters
        ----------
        box_list : Optional[list[str]], optional
            List of boxes to configure. Defaults to None.

        Examples
        --------
        >>> from qubex import Experiment
        >>> exp = Experiment(chip_id="64Q")
        >>> exp.tools.configure_boxes()
        """
        self._exp._config.configure_box_settings(self._exp._chip_id, include=box_list)

    def print_wiring_info(self):
        """
        Print the wiring information of the chip.

        Examples
        --------
        >>> from qubex import Experiment
        >>> exp = Experiment(chip_id="64Q")
        >>> exp.tools.print_wiring_info()
        """

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"WIRING INFO ({self._exp._chip_id})",
        )
        table.add_column("QUBIT", justify="center", width=7)
        table.add_column("CTRL", justify="center", width=11)
        table.add_column("READ.OUT", justify="center", width=11)
        table.add_column("READ.IN", justify="center", width=11)

        for qubit in self._exp.qubits:
            ports = self._exp._config.get_ports_by_qubit(
                chip_id=self._exp._chip_id,
                qubit=qubit.label,
            )
            ctrl_port = ports[0]
            read_out_port = ports[1]
            read_in_port = ports[2]
            if ctrl_port is None or read_out_port is None or read_in_port is None:
                table.add_row(qubit.label, "-", "-", "-")
                continue
            ctrl_box = ctrl_port.box
            read_out_box = read_out_port.box
            read_in_box = read_in_port.box
            ctrl = f"{ctrl_box.id}-{ctrl_port.number}"
            read_out = f"{read_out_box.id}-{read_out_port.number}"
            read_in = f"{read_in_box.id}-{read_in_port.number}"
            table.add_row(qubit.label, ctrl, read_out, read_in)

        console.print(table)

    def print_box_info(self, box_id: str) -> None:
        """
        Print the information of a box.

        Parameters
        ----------
        box_id : str
            Identifier of the box.

        Examples
        --------
        >>> from qubex import Experiment
        >>> exp = Experiment(chip_id="64Q")
        >>> exp.tools.print_box_info("Q73A")
        """
        if box_id not in self._exp.available_boxes:
            console.print(
                f"Box {box_id} not in available boxes: {self._exp.available_boxes}"
            )
            return

        table1 = Table(
            show_header=True,
            header_style="bold",
            title=f"BOX INFO ({box_id})",
        )
        table2 = Table(
            show_header=True,
            header_style="bold",
        )
        table1.add_column("PORT", justify="right")
        table1.add_column("TYPE", justify="right")
        table1.add_column("SSB", justify="right")
        table1.add_column("LO", justify="right")
        table1.add_column("CNCO", justify="right")
        table1.add_column("FSC", justify="right")
        table2.add_column("PORT", justify="right")
        table2.add_column("TYPE", justify="right")
        table2.add_column("FNCO-0", justify="right")
        table2.add_column("FNCO-1", justify="right")
        table2.add_column("FNCO-2", justify="right")
        table2.add_column("FNCO-3", justify="right")

        port_map = self._exp._config.get_port_map(box_id)
        ssb_map = {"U": "[cyan]USB[/cyan]", "L": "[green]LSB[/green]"}

        ports = self.dump_box(box_id)["ports"]
        for number, port in ports.items():
            direction = port["direction"]
            lo = int(port["lo_freq"])
            cnco = int(port["cnco_freq"])
            type = port_map[number].value
            if direction == "in":
                ssb = ""
                fsc = ""
                fncos = [str(int(ch["fnco_freq"])) for ch in port["runits"].values()]
            elif direction == "out":
                ssb = ssb_map[port["sideband"]]
                fsc = port["fullscale_current"]
                fncos = [str(int(ch["fnco_freq"])) for ch in port["channels"].values()]
            table1.add_row(str(number), type, ssb, str(lo), str(cnco), str(fsc))
            table2.add_row(str(number), type, *fncos)
        console.print(table1)
        console.print(table2)
