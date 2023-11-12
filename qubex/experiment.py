import os, datetime, pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .qube_manager import QubeManager
from .pulse import Rect, Waveform, PulseSequence
from .analysis import rotate, get_angle, fit_rabi
from .plot import show_pulse_sequences, show_measurement_results
from .typing import (
    QubitKey,
    QubitDict,
    IQValue,
    IQArray,
    ReadoutPorts,
    ParametricWaveform,
)
from .params import (
    ampl_hpi_dict,
)
from .consts import (
    T_CONTROL,
    T_READOUT,
    READOUT_RANGE,
)


@dataclass
class ExperimentResult:
    qubit: QubitKey
    sweep_range: NDArray
    data: IQArray
    phase_shift: float
    datetime: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def rotated(self) -> IQArray:
        return self.data * np.exp(-1j * self.phase_shift)


@dataclass
class RabiParams:
    qubit: QubitKey
    phase_shift: float
    amplitude: float
    omega: float
    phi: float
    offset: float


class Experiment:
    def __init__(
        self,
        qube_id: str,
        mux_number: int,
        readout_ports: ReadoutPorts = ("port0", "port1"),
        repeats: int = 10_000,
        interval: int = 150_000,
        ctrl_duration: int = T_CONTROL,
        data_dir="./data",
    ):
        self.qube = QubeManager(
            qube_id=qube_id,
            mux_number=mux_number,
            readout_ports=readout_ports,
            repeats=repeats,
            interval=interval,
            ctrl_duration=ctrl_duration,
        )
        self.data_dir = data_dir

    def save_data(self, data: object, name: str = "data"):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{current_time}_{name}.pkl"
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {file_path}")

    def load_data(self, name: str):
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        path = os.path.join(self.data_dir, name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def measure(
        self,
        waveforms: QubitDict[Waveform],
    ) -> QubitDict[IQValue]:
        qubits = list(waveforms.keys())
        result = self.qube.measure(
            ctrl_qubits=qubits,
            read_qubits=qubits,
            waveforms=waveforms,
        )
        return result

    def rabi_experiment(
        self,
        amplitudes: QubitDict[float],
        time_range=np.arange(0, 201, 10),
    ) -> QubitDict[ExperimentResult]:
        qubits = list(amplitudes.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for idx, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                )
                for qubit in qubits
            }

            response = self.measure(waveforms)

            for qubit, iq in response.items():
                signals[qubit].append(iq)

            self.show_experiments_results(
                control_qubits=control_qubits,
                readout_qubits=readout_qubits,
                sweep_range=time_range,
                idx=idx,
                signals=signals,
            )

        phase_shifts = {qubit: get_angle(data) for qubit, data in signals.items()}

        result = {
            qubit: ExperimentResult(
                qubit=qubit,
                sweep_range=time_range,
                data=np.array(data),
                phase_shift=phase_shifts[qubit],
            )
            for qubit, data in signals.items()
        }
        return result

    def sweep_pramameter(
        self,
        sweep_range: NDArray,
        waveforms: QubitDict[ParametricWaveform],
        pulse_count=1,
    ) -> QubitDict[ExperimentResult]:
        qubits = list(waveforms.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for idx, var in enumerate(sweep_range):
            waveforms_var = {
                qubit: PulseSequence([waveform(var)] * pulse_count)
                for qubit, waveform in waveforms.items()
            }

            response = self.measure(waveforms_var)

            for qubit, iq in response.items():
                signals[qubit].append(iq)

            self.show_experiments_results(
                control_qubits=control_qubits,
                readout_qubits=readout_qubits,
                sweep_range=sweep_range,
                idx=idx,
                signals=signals,
            )

        phase_shifts = {qubit: get_angle(data) for qubit, data in signals.items()}

        result = {
            qubit: ExperimentResult(
                qubit=qubit,
                sweep_range=sweep_range,
                data=np.array(data),
                phase_shift=phase_shifts[qubit],
            )
            for qubit, data in signals.items()
        }
        return result

    def rabi_check(
        self,
        time_range=np.arange(0, 201, 10),
    ) -> QubitDict[ExperimentResult]:
        amplitudes = ampl_hpi_dict[self.qube.qube_id]
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
        )
        return result

    def repeat_pulse(
        self,
        waveforms: QubitDict[Waveform],
        n: int,
    ) -> QubitDict[ExperimentResult]:
        result = self.sweep_pramameter(
            sweep_range=np.arange(n + 1),
            waveforms={
                qubit: lambda x: PulseSequence([waveform] * int(x))
                for qubit, waveform in waveforms.items()
            },
            pulse_count=1,
        )
        return result

    def show_experiments_results(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        sweep_range: NDArray,
        idx: int,
        signals: QubitDict[list[IQValue]],
    ):
        signals_rotated = {}

        for qubit in signals:
            angle = get_angle(signals[qubit])
            signals_rotated[qubit] = rotate(signals[qubit], angle)

        control_waveforms = self.qube.get_control_waveforms(control_qubits)
        control_times = self.qube.get_control_times(control_qubits)
        readout_tx_waveforms = self.qube.get_readout_tx_waveforms(readout_qubits)
        readout_tx_times = self.qube.get_readout_tx_times(readout_qubits)
        readout_rx_waveforms = self.qube.get_readout_rx_waveforms(readout_qubits)
        readout_rx_times = self.qube.get_readout_rx_times(readout_qubits)

        clear_output(True)
        show_measurement_results(
            readout_qubits=readout_qubits,
            readout_rx_waveforms=readout_rx_waveforms,
            readout_rx_times=readout_rx_times,
            sweep_range=sweep_range[: idx + 1],
            signals=signals,
            signals_rotated=signals_rotated,
            readout_range=READOUT_RANGE,
        )
        show_pulse_sequences(
            control_qubits=control_qubits,
            control_waveforms=control_waveforms,
            control_times=control_times,
            control_duration=self.qube.ctrl_duration_,
            readout_tx_qubits=readout_qubits,
            readout_tx_waveforms=readout_tx_waveforms,
            readout_tx_times=readout_tx_times,
            readout_tx_duration=T_READOUT,
        )
        print(f"{idx+1}/{len(sweep_range)}")

    def fit_rabi_params(
        self,
        result: ExperimentResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Normalize the Rabi oscillation data.
        """
        time = result.sweep_range

        # Rotate the data to the vertical (Q) axis
        angle = get_angle(data=result.data)
        points = rotate(data=result.data, angle=angle)
        values = points.imag
        print(f"Phase shift: {angle:.3f} rad, {angle * 180 / np.pi:.3f} deg")

        # Estimate the initial parameters
        omega0 = 2 * np.pi / (time[-1] - time[0])
        ampl_est = (np.max(values) - np.min(values)) / 2
        omega_est = wave_count * omega0
        phase_est = np.pi
        offset_est = (np.max(values) + np.min(values)) / 2
        p0 = (ampl_est, omega_est, phase_est, offset_est)

        # Set the bounds for the parameters
        bounds = (
            [0, 0, -np.pi, -2 * np.abs(offset_est)],
            [2 * ampl_est, 2 * omega_est, np.pi, 2 * np.abs(offset_est)],
        )

        # Fit the data
        popt, _ = fit_rabi(time, values, p0, bounds)

        # Set the parameters as the instance attributes
        rabi_params = RabiParams(
            qubit=result.qubit,
            phase_shift=angle,
            amplitude=popt[0],
            omega=popt[1],
            phi=popt[2],
            offset=popt[3],
        )
        return rabi_params

    def expectation_values(
        self,
        result: ExperimentResult,
        params: RabiParams,
    ) -> NDArray[np.float64]:
        values = result.rotated.imag
        values_normalized = -(values - params.offset) / params.amplitude
        return values_normalized

    def expectation_value(
        self,
        iq: IQValue,
        params: RabiParams,
    ) -> float:
        iq = iq * np.exp(-1j * params.phase_shift)
        value = iq.imag
        value = -(value - params.offset) / params.amplitude
        return value

    def plot_expectation_values(
        self,
        result: ExperimentResult,
        params: RabiParams,
        title: str = "Expectation value",
        xlabel: str = "Time / ns",
        ylabel: str = r"Expectation Value $\langle \sigma_z \rangle$",
    ) -> NDArray[np.float64]:
        values = self.expectation_values(result, params)
        _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(result.sweep_range, values, "o-")
        ax.set_title(f"{title}, {result.qubit}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

        return values
