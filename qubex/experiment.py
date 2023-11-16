import os
import datetime
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray
from IPython.display import clear_output

from .qube_manager import QubeManager
from .pulse import Rect, Waveform
from .analysis import fit_and_rotate, get_angle, fit_rabi
from .visualization import show_pulse_sequences, show_measurement_results
from .typing import (
    QubitKey,
    QubitDict,
    IQValue,
    IQArray,
    IntArray,
    ReadoutPorts,
    ParametricWaveform,
)
from .consts import (
    T_CONTROL,
    T_READOUT,
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
    fluctuation: float
    amplitude: float
    omega: float
    phi: float
    offset: float


class Experiment:
    def __init__(
        self,
        qube_id: str,
        mux_number: int,
        params: str,
        readout_ports: ReadoutPorts = ("port0", "port1"),
        control_duration: int = T_CONTROL,
        readout_duration: int = T_READOUT,
        repeats: int = 10_000,
        interval: int = 150_000,
        data_path="./data",
    ):
        self.params = self._get_params(params, qube_id)
        self.qube_manager: Final = QubeManager(
            qube_id=qube_id,
            mux_number=mux_number,
            params=self.params,
            readout_ports=readout_ports,
            control_duration=control_duration,
            readout_duration=readout_duration,
        )
        self.qube: Final = self.qube_manager.qube
        self.repeats: Final = repeats
        self.interval: Final = interval
        self.data_path: Final = data_path

    def _get_params(self, cooldown: str, qube_id: str) -> dict:
        current_dir = os.path.dirname(__file__)
        params_path = os.path.join(current_dir, "params", cooldown, f"{qube_id}.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return params

    def save_data(self, data: object, name: str):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        extension = ".pkl"
        counter = 1
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(
            self.data_path,
            f"{current_date}_{name}_{counter}{extension}",
        )

        # Check if the file exists and create a new name if it does
        while os.path.exists(file_path):
            file_path = os.path.join(
                self.data_path,
                f"{current_date}_{name}_{counter}{extension}",
            )
            counter += 1

        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {file_path}")

    def load_data(self, name: str):
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        path = os.path.join(self.data_path, name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def loopback_mode(self, use_loopback: bool):
        self.qube_manager.loopback_mode(use_loopback)

    def get_control_frequency(self, qubit: QubitKey) -> float:
        return self.qube_manager.get_control_frequency(qubit)

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        self.qube_manager.set_control_frequency(qubit, frequency)

    def measure(
        self,
        waveforms: QubitDict[Waveform],
        repeats: Optional[int] = None,
        interval: Optional[int] = None,
    ) -> QubitDict[IQValue]:
        if repeats is None:
            repeats = self.repeats
        if interval is None:
            interval = self.interval
        qubits = list(waveforms.keys())
        result = self.qube_manager.measure(
            control_qubits=qubits,
            readout_qubits=qubits,
            control_waveforms=waveforms,
            repeats=repeats,
            interval=interval,
        )
        return result

    def rabi_experiment(
        self,
        time_range: IntArray,
        amplitudes: QubitDict[float],
    ) -> QubitDict[ExperimentResult]:
        qubits = list(amplitudes.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                )
                for qubit in qubits
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            self.show_experiment_results(
                control_qubits=control_qubits,
                readout_qubits=readout_qubits,
                sweep_range=time_range,
                index=index,
                signals=signals,
            )

        result = {
            qubit: ExperimentResult(
                qubit=qubit,
                sweep_range=time_range,
                data=np.array(data),
                phase_shift=get_angle(data),
            )
            for qubit, data in signals.items()
        }
        return result

    def sweep_parameter(
        self,
        sweep_range: NDArray,
        parametric_waveforms: QubitDict[ParametricWaveform],
        pulse_count=1,
    ) -> QubitDict[ExperimentResult]:
        qubits = list(parametric_waveforms.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, param in enumerate(sweep_range):
            waveforms = {
                qubit: waveform(param).repeated(pulse_count)
                for qubit, waveform in parametric_waveforms.items()
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            self.show_experiment_results(
                control_qubits=control_qubits,
                readout_qubits=readout_qubits,
                sweep_range=sweep_range,
                index=index,
                signals=signals,
            )

        result = {
            qubit: ExperimentResult(
                qubit=qubit,
                sweep_range=sweep_range,
                data=np.array(data),
                phase_shift=get_angle(data),
            )
            for qubit, data in signals.items()
        }
        return result

    def check_rabi(
        self,
        time_range=np.arange(0, 201, 10),
    ) -> QubitDict[ExperimentResult]:
        amplitudes = self.params["hpi_amplitude"]
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
        parametric_waveforms = {
            qubit: lambda param, w=waveform: w.repeated(int(param))
            for qubit, waveform in waveforms.items()
        }
        result = self.sweep_parameter(
            sweep_range=np.arange(n + 1),
            parametric_waveforms=parametric_waveforms,
            pulse_count=1,
        )
        return result

    def show_experiment_results(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        sweep_range: NDArray,
        index: int,
        signals: QubitDict[list[IQValue]],
    ):
        signals_rotated = {
            qubit: fit_and_rotate(data) for qubit, data in signals.items()
        }

        ctrl_waveforms = self.qube_manager.get_control_waveforms(control_qubits)
        ctrl_times = self.qube_manager.get_control_times(control_qubits)
        rotx_waveforms = self.qube_manager.get_readout_tx_waveforms(readout_qubits)
        rotx_times = self.qube_manager.get_readout_tx_times(readout_qubits)
        rorx_waveforms = self.qube_manager.get_readout_rx_waveforms(readout_qubits)
        rorx_times = self.qube_manager.get_readout_rx_times(readout_qubits)
        readout_range = self.qube_manager.readout_range()
        control_duration = self.qube_manager.control_duration
        readout_duration = self.qube_manager.readout_duration

        clear_output(True)
        show_measurement_results(
            qubits=readout_qubits,
            waveforms=rorx_waveforms,
            times=rorx_times,
            sweep_range=sweep_range[: index + 1],
            signals=signals,
            signals_rotated=signals_rotated,
            readout_range=readout_range,
        )
        show_pulse_sequences(
            control_qubits=control_qubits,
            control_waveforms=ctrl_waveforms,
            control_times=ctrl_times,
            control_duration=control_duration,
            readout_qubits=readout_qubits,
            readout_waveforms=rotx_waveforms,
            readout_times=rotx_times,
            readout_duration=readout_duration,
        )
        print(f"{index+1}/{len(sweep_range)}")

    def fit_rabi(
        self,
        experiment_result: ExperimentResult,
        wave_count=2.5,
    ) -> RabiParams:
        times = experiment_result.sweep_range
        data = experiment_result.data

        phase_shift, fluctuation, popt = fit_rabi(
            times=times,
            data=data,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=experiment_result.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[1],
            phi=popt[2],
            offset=popt[3],
        )
        return rabi_params

    def expectation_values(
        self,
        experiment_result: ExperimentResult,
        rabi_params: RabiParams,
    ) -> NDArray[np.float64]:
        values = experiment_result.rotated.imag
        values_normalized = -(values - rabi_params.offset) / rabi_params.amplitude
        return values_normalized

    def expectation_value(
        self,
        iq_value: IQValue,
        rabi_params: RabiParams,
    ) -> float:
        iq_value = iq_value * np.exp(-1j * rabi_params.phase_shift)
        value = iq_value.imag
        value = -(value - rabi_params.offset) / rabi_params.amplitude
        return value
