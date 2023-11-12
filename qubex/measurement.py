import os, datetime, pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from IPython.display import clear_output

import qubecalib as qc
from qubecalib.pulse import Schedule, Channel, Blank, Arbitrary, Read
from qubecalib.setupqube import run

qc.ui.MATPLOTLIB_PYPLOT = plt  # type: ignore

from .pulse import Rect, Waveform, PulseSequence
from .analysis import rotate, get_angle, fit_rabi
from .plot import show_pulse_sequences, show_measurement_results
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
    ctrl_freq_dict,
    ro_freq_dict,
    ro_ampl_dict,
    ampl_hpi_dict,
)

SAMPLING_PERIOD: int = 2  # [ns]
MIN_SAMPLE: int = 64  # min number of samples of e7awg
MIN_DURATION: int = MIN_SAMPLE * SAMPLING_PERIOD
T_CONTROL: int = 10 * 1024  # [ns]
T_READOUT: int = 1024  # [ns]
T_MARGIN: int = MIN_DURATION  # [ns]
READOUT_RANGE = slice(T_MARGIN // 2, (T_READOUT + T_MARGIN) // 2)

CTRL_HI = "_hi"
CTRL_LO = "_lo"
READ_TX = "TX_"
READ_RX = "RX_"

MUX = [[f"Q{i*4+j:02d}" for j in range(4)] for i in range(16)]


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


class Measurement:
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
        self.qube_id = qube_id
        self.qube = qc.ui.QubeControl(f"{qube_id}.yml").qube
        self.qubits = MUX[mux_number]
        self.readout_ports = readout_ports
        self.repeats = repeats
        self.interval = interval
        self.ctrl_duration = ctrl_duration
        self.schedule = Schedule()
        self.data_dir = data_dir
        self._init_channels()
        self._init_ports(self.qube)

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

    def get_control_frequency(self, qubit: QubitKey) -> float:
        return self.schedule[qubit].center_frequency

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        self.schedule[qubit].center_frequency = frequency

    def _ctrl_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit]

    def _read_tx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READ_TX + qubit]

    def _read_rx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READ_RX + qubit]

    def _ctrl_channels(self) -> list[Channel]:
        return [self._ctrl_channel(qubit) for qubit in self.all_ctrl_qubits]

    def _read_tx_channels(self) -> list[Channel]:
        return [self._read_tx_channel(qubit) for qubit in self.all_read_qubits]

    def _read_rx_channels(self) -> list[Channel]:
        return [self._read_rx_channel(qubit) for qubit in self.all_read_qubits]

    def _ctrl_slot(self, qubit: QubitKey) -> Arbitrary:
        return self._ctrl_channel(qubit).findall(Arbitrary)[0]

    def _read_tx_slot(self, qubit: QubitKey) -> Arbitrary:
        return self._read_tx_channel(qubit).findall(Arbitrary)[0]

    def _read_rx_slot(self, qubit: QubitKey) -> Read:
        return self._read_rx_channel(qubit).findall(Read)[0]

    def _ctrl_times(self, qubit: QubitKey) -> IntArray:
        channel = self._ctrl_channel(qubit)
        slot = self._ctrl_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _read_times(self, qubit: QubitKey) -> IntArray:
        channel = self._read_tx_channel(qubit)
        slot = self._read_tx_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _init_channels(self):
        self.all_read_qubits = self.qubits
        self.all_ctrl_qubits = []

        for qubit in self.qubits:
            self.all_ctrl_qubits.extend([qubit + CTRL_LO, qubit, qubit + CTRL_HI])

        for qubit in self.all_read_qubits:
            self.schedule[READ_TX + qubit] = Channel(
                center_frequency=ro_freq_dict[self.qube_id][qubit],
            )
            self.schedule[READ_RX + qubit] = Channel(
                center_frequency=ro_freq_dict[self.qube_id][qubit],
            )

        for qubit in self.all_ctrl_qubits:
            self.schedule[qubit] = Channel(
                center_frequency=ctrl_freq_dict[qubit],
            )

    def _init_ports(self, qube):
        port_tx = qube.ports[self.readout_ports[0]]
        port_rx = qube.ports[self.readout_ports[1]]
        self.port_tx = port_tx
        self.port_rx = port_rx

        port_tx.lo.mhz = 11500
        port_tx.nco.mhz = 1500
        port_tx.mix.ssb = qc.qube.SSB.LSB
        port_tx.awg0.nco.mhz = 0
        port_tx.mix.vatt = 0x800

        port_rx.nco.mhz = qube.port0.nco.mhz
        port_rx.adc.capt0.ssb = qc.qube.SSB.LSB
        port_rx.delay = 128 + 6 * 128  # [ns]

        qube.port5.lo.mhz = 9500
        qube.port5.nco.mhz = 1875
        qube.port5.awg0.nco.mhz = 0
        qube.port5.awg1.nco.mhz = 0
        qube.port5.awg2.nco.mhz = 0
        qube.port5.mix.vatt = 0x800

        qube.port6.lo.mhz = 10000
        qube.port6.nco.mhz = 1500
        qube.port6.awg0.nco.mhz = 0
        qube.port6.awg1.nco.mhz = 0
        qube.port6.awg2.nco.mhz = 0
        qube.port6.mix.vatt = 0x800

        qube.port7.lo.mhz = 9500
        qube.port7.nco.mhz = 1125
        qube.port7.awg0.nco.mhz = 0
        qube.port7.awg1.nco.mhz = 0
        qube.port7.awg2.nco.mhz = 0
        qube.port7.mix.vatt = 0x800

        qube.port8.lo.mhz = 9000
        qube.port8.nco.mhz = 1875
        qube.port8.awg0.nco.mhz = 0
        qube.port8.awg1.nco.mhz = 0
        qube.port8.awg2.nco.mhz = 0
        qube.port8.mix.vatt = 0x800

        self.triggers = [port_tx.dac.awg0]

        self.adda_to_channels = {
            port_tx.dac.awg0: self._read_tx_channels(),
            port_rx.adc.capt0: self._read_rx_channels(),
            qube.port5.dac.awg0: [self._ctrl_channel(self.qubits[0] + CTRL_LO)],
            qube.port6.dac.awg0: [self._ctrl_channel(self.qubits[1] + CTRL_LO)],
            qube.port7.dac.awg0: [self._ctrl_channel(self.qubits[2] + CTRL_LO)],
            qube.port8.dac.awg0: [self._ctrl_channel(self.qubits[3] + CTRL_LO)],
            qube.port5.dac.awg1: [self._ctrl_channel(self.qubits[0])],
            qube.port6.dac.awg1: [self._ctrl_channel(self.qubits[1])],
            qube.port7.dac.awg1: [self._ctrl_channel(self.qubits[2])],
            qube.port8.dac.awg1: [self._ctrl_channel(self.qubits[3])],
            qube.port5.dac.awg2: [self._ctrl_channel(self.qubits[0] + CTRL_HI)],
            qube.port6.dac.awg2: [self._ctrl_channel(self.qubits[1] + CTRL_HI)],
            qube.port7.dac.awg2: [self._ctrl_channel(self.qubits[2] + CTRL_HI)],
            qube.port8.dac.awg2: [self._ctrl_channel(self.qubits[3] + CTRL_HI)],
        }

    def set_waveforms(
        self,
        ctrl_qubits: list[QubitKey],
        read_qubits: list[QubitKey],
        waveforms: QubitDict[IQArray],
    ):
        max_length = max([len(waveform) for waveform in waveforms.values()])
        max_duration = max_length * SAMPLING_PERIOD
        # TODO: 動的に ctrl_duration を決めるためには、位相のずれを考慮する必要がある
        # self.schedule.offset = (max_duration // MIN_DURATION + 1) * MIN_DURATION + T_MARGIN
        self.ctrl_duration_ = (
            max_duration // MIN_DURATION + 1
        ) * MIN_DURATION + T_MARGIN

        self.schedule.offset = self.ctrl_duration

        for ch in self._ctrl_channels():
            ch.clear()
            ch.append(Arbitrary(duration=self.schedule.offset, amplitude=1))
            ch.append(Blank(duration=T_READOUT + 4 * T_MARGIN))

        for ch in self._read_tx_channels():
            ch.clear()
            ch.append(Blank(duration=self.schedule.offset))
            ch.append(Arbitrary(duration=T_READOUT, amplitude=1))
            ch.append(Blank(duration=4 * T_MARGIN))

        for ch in self._read_rx_channels():
            ch.clear()
            ch.append(Blank(duration=self.schedule.offset - T_MARGIN))
            ch.append(Read(duration=T_READOUT + 5 * T_MARGIN))

        durations = [channel.duration for channel in self.schedule.values()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

        for qubit in ctrl_qubits:
            T = len(waveforms[qubit]) * SAMPLING_PERIOD
            t = self._ctrl_times(qubit)
            self._ctrl_slot(qubit).iq[(-T <= t) & (t < 0)] = waveforms[qubit]

        for qubit in read_qubits:
            ampl = ro_ampl_dict[self.qube_id][qubit]
            waveform = Rect(
                duration=T_READOUT,
                amplitude=ampl,
                risetime=50,
            )
            self._read_tx_slot(qubit).iq[:] = waveform.values

    def get_control_waveforms(
        self,
        ctrl_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._ctrl_slot(qubit).iq for qubit in ctrl_qubits}

    def get_control_times(
        self,
        ctrl_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._ctrl_times(qubit) for qubit in ctrl_qubits}

    def get_readout_tx_waveforms(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._read_tx_slot(qubit).iq for qubit in read_qubits}

    def get_readout_tx_times(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._read_times(qubit) for qubit in read_qubits}

    def get_readout_rx_waveforms(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._read_rx_slot(qubit).iq for qubit in read_qubits}  # type: ignore

    def get_readout_rx_times(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._read_rx_slot(qubit).timestamp for qubit in read_qubits}

    def _integrated_iq(
        self,
        waveforms: QubitDict[IQArray],
    ) -> QubitDict[IQValue]:
        iq = {
            qubit: waveform[READOUT_RANGE].mean()
            for qubit, waveform in waveforms.items()
        }
        return iq

    def _measure(
        self,
        ctrl_qubits: list[QubitKey],
        read_qubits: list[QubitKey],
        waveforms: QubitDict[Waveform],
    ) -> QubitDict[IQValue]:
        self.set_waveforms(
            ctrl_qubits=ctrl_qubits,
            read_qubits=read_qubits,
            waveforms={qubit: waveform.values for qubit, waveform in waveforms.items()},
        )
        run(
            self.schedule,
            repeats=self.repeats,
            interval=self.interval,
            adda_to_channels=self.adda_to_channels,
            triggers=self.triggers,
        )
        rx_waveforms = self.get_readout_rx_waveforms(read_qubits)
        result = self._integrated_iq(rx_waveforms)
        return result

    def measure(
        self,
        waveforms: QubitDict[Waveform],
    ) -> QubitDict[IQValue]:
        qubits = list(waveforms.keys())
        result = self._measure(
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

        control_waveforms = self.get_control_waveforms(control_qubits)
        control_times = self.get_control_times(control_qubits)
        readout_tx_waveforms = self.get_readout_tx_waveforms(readout_qubits)
        readout_tx_times = self.get_readout_tx_times(readout_qubits)
        readout_rx_waveforms = self.get_readout_rx_waveforms(readout_qubits)
        readout_rx_times = self.get_readout_rx_times(readout_qubits)

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
            control_duration=self.ctrl_duration_,
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
