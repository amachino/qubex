import os
import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray
from IPython.display import clear_output

from .qube_manager import QubeManager
from .pulse import Rect, Waveform
from .analysis import fit_and_rotate, rotate, get_angle, fit_rabi
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
        control_window: int = T_CONTROL,
        measurement_repetition: int = 10_000,
        measurement_inverval: int = 150_000,
        data_path="./data",
    ):
        self.qube_manager: Final = QubeManager(
            qube_id=qube_id,
            mux_number=mux_number,
            readout_ports=readout_ports,
            control_window=control_window,
        )
        self.qube_id: Final = qube_id
        self.qube: Final = self.qube_manager.qube
        self.measurement_repetition: Final = measurement_repetition
        self.measurement_inverval: Final = measurement_inverval
        self.data_path: Final = data_path

    def save_data(self, data: object, name: str = "data"):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{current_time}_{name}.pkl"
        file_path = os.path.join(self.data_path, file_name)
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

    def measure(
        self,
        waveforms: QubitDict[Waveform],
        repeats: Optional[int] = None,
        interval: Optional[int] = None,
    ) -> QubitDict[IQValue]:
        if repeats is None:
            repeats = self.measurement_repetition
        if interval is None:
            interval = self.measurement_inverval
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
        amplitudes: QubitDict[float],
        time_range: IntArray,
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
        waveform_repetition=1,
    ) -> QubitDict[ExperimentResult]:
        qubits = list(parametric_waveforms.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, param in enumerate(sweep_range):
            waveforms = {
                qubit: waveform(param).repeated(waveform_repetition)
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

    def rabi_check(
        self,
        time_range=np.arange(0, 201, 10),
    ) -> QubitDict[ExperimentResult]:
        amplitudes = ampl_hpi_dict[self.qube_id]
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
            waveform_repetition=1,
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

        clear_output(True)
        show_measurement_results(
            qubits=readout_qubits,
            waveforms=rorx_waveforms,
            times=rorx_times,
            sweep_range=sweep_range[: index + 1],
            signals=signals,
            signals_rotated=signals_rotated,
            readout_range=READOUT_RANGE,
        )
        show_pulse_sequences(
            control_qubits=control_qubits,
            control_waveforms=ctrl_waveforms,
            control_times=ctrl_times,
            control_duration=T_CONTROL,
            readout_qubits=readout_qubits,
            readout_waveforms=rotx_waveforms,
            readout_times=rotx_times,
            readout_duration=T_READOUT,
        )
        print(f"{index+1}/{len(sweep_range)}")

    def fit_rabi_params(
        self,
        result: ExperimentResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Normalize the Rabi oscillation data.
        """
        times = result.sweep_range

        # Rotate the data to the vertical (Q) axis
        angle = get_angle(data=result.data)
        points = rotate(data=result.data, angle=angle)
        values = points.imag
        print(f"Phase shift: {angle:.3f} rad, {angle * 180 / np.pi:.3f} deg")

        # Estimate the initial parameters
        omega0 = 2 * np.pi / (times[-1] - times[0])
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
        popt, _ = fit_rabi(times, values, p0, bounds)

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
