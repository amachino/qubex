from attr import dataclass
from typing import Callable, Optional

from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.typing import NDArray

import qubecalib as qc
from qubecalib.pulse import Schedule, Channel, Blank, Arbitrary, Read
from qubecalib.setupqube import run

qc.ui.MATPLOTLIB_PYPLOT = plt

from .pulse import Rect, Waveform, Sequence
from .analysis import rotate, fit_and_rotate, fit_rabi

from .params import (
    ctrl_freq_dict,
    ro_freq_dict,
    ro_ampl_dict,
    ampl_hpi_dict,
)

SAMPLING_PERIOD = 2  # [ns]
MIN_SAMPLE = 64
MIN_DURATION = MIN_SAMPLE * SAMPLING_PERIOD
T_CONTROL = 10 * 2048
T_READOUT = 12 * MIN_DURATION
T_MARGIN = 2 * MIN_DURATION
READOUT_RANGE = slice(200, int(200 + 800 / 1.5))

CTRL_HI = "_hi"
CTRL_LO = "_lo"
READ_TX = "TX_"
READ_RX = "RX_"

MUX = [
    ["Q00", "Q01", "Q02", "Q03"],
    ["Q04", "Q05", "Q06", "Q07"],
    ["Q08", "Q09", "Q10", "Q11"],
    ["Q12", "Q13", "Q14", "Q15"],
]


@dataclass
class RabiParams:
    qubit: str
    phase_shift: float
    amplitude: float
    omega: float
    phi: float
    offset: float


@dataclass
class MeasurementResult:
    qubit: str
    sweep_range: NDArray
    data: NDArray[np.complex128]
    phase_shift: float

    @property
    def real(self):
        return np.real(self.data)

    @property
    def imag(self):
        return np.imag(self.data)

    @property
    def vector(self):
        return np.column_stack([self.real, self.imag])

    @property
    def rotated(self):
        return self.data * np.exp(-1j * self.phase_shift)


class Measurement:
    def __init__(
        self,
        qube_id: str,
        mux_number: int,
        readout_ports: tuple[str, str] = ("port0", "port1"),
        repeats=10_000,
        interval=150_000,
        read_range=READOUT_RANGE,
    ):
        self.qube_id = qube_id
        self.qube = qc.ui.QubeControl(f"{qube_id}.yml").qube
        self.qubits = MUX[mux_number]
        self.readout_ports = readout_ports
        self.repeats = repeats
        self.interval = interval
        self.ctrl_duration = T_CONTROL
        self.read_range = read_range
        self.schedule = Schedule()
        self.rabi_params: dict[str, RabiParams] = {}
        self._setup(self.qube)

    def ctrl_frequency(self, qubit: str) -> float:
        return self.schedule[qubit].center_frequency

    def set_ctrl_frequency(self, qubit: str, frequency: float):
        self.schedule[qubit].center_frequency = frequency

    def _ctrl_channel(self, qubit: str) -> Channel:
        return self.schedule[qubit]

    def _read_tx_channel(self, qubit: str) -> Channel:
        return self.schedule[READ_TX + qubit]

    def _read_rx_channel(self, qubit: str) -> Channel:
        return self.schedule[READ_RX + qubit]

    def _ctrl_channels(self) -> list[Channel]:
        return [self._ctrl_channel(qubit) for qubit in self.all_ctrl_qubits]

    def _read_tx_channels(self) -> list[Channel]:
        return [self._read_tx_channel(qubit) for qubit in self.all_read_qubits]

    def _read_rx_channels(self) -> list[Channel]:
        return [self._read_rx_channel(qubit) for qubit in self.all_read_qubits]

    def _ctrl_slot(self, qubit: str) -> Arbitrary:
        return self._ctrl_channel(qubit).findall(Arbitrary)[0]

    def _read_tx_slot(self, qubit: str) -> Arbitrary:
        return self._read_tx_channel(qubit).findall(Arbitrary)[0]

    def _read_rx_slot(self, qubit: str) -> Read:
        return self._read_rx_channel(qubit).findall(Read)[0]

    def _ctrl_times(self, qubit: str) -> np.ndarray:
        channel = self._ctrl_channel(qubit)
        slot = self._ctrl_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _read_times(self, qubit: str) -> np.ndarray:
        channel = self._read_tx_channel(qubit)
        slot = self._read_tx_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _setup_channels(self):
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

    def _setup(self, qube):
        self._setup_channels()

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

    def set_circuit(
        self,
        ctrl_qubits: list[str],
        read_qubits: list[str],
        waveforms: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        # TODO: 動的に ctrl_duration を決めるためには、位相のずれを考慮する必要がある
        # max_length = max([len(waveform) for waveform in waveforms.values()])
        # max_duration = max_length * SAMPLING_PERIOD
        # ctrl_duration = (max_duration // MIN_DURATION + 1) * MIN_DURATION + T_MARGIN
        ctrl_duration = self.ctrl_duration

        self.schedule.offset = ctrl_duration  # [ns]

        for ch in self._ctrl_channels():
            ch.clear()
            ch.append(Arbitrary(duration=ctrl_duration, amplitude=1))
            ch.append(Blank(duration=T_READOUT + 4 * T_MARGIN))

        for ch in self._read_tx_channels():
            ch.clear()
            ch.append(Blank(duration=ctrl_duration))
            ch.append(Arbitrary(duration=T_READOUT, amplitude=1))
            ch.append(Blank(duration=4 * T_MARGIN))

        for ch in self._read_rx_channels():
            ch.clear()
            ch.append(Blank(duration=ctrl_duration - T_MARGIN))
            ch.append(Read(duration=T_READOUT + 5 * T_MARGIN))

        durations = [channel.duration for channel in self.schedule.values()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

        ctrl_waveforms = {}
        for qubit in ctrl_qubits:
            T = len(waveforms[qubit]) * SAMPLING_PERIOD
            t = self._ctrl_times(qubit)
            self._ctrl_slot(qubit).iq[(-T <= t) & (t < 0)] = waveforms[qubit]
            iq = self._ctrl_slot(qubit).iq
            ctrl_waveforms[qubit] = iq

        read_waveforms = {}
        for qubit in read_qubits:
            ampl = ro_ampl_dict[self.qube_id][qubit]
            t = self._read_times(qubit)
            self._read_tx_slot(qubit).iq[:] = ampl * self._readout_waveform(t)
            iq = self._read_tx_slot(qubit).iq
            read_waveforms[qubit] = iq

        return ctrl_waveforms, read_waveforms

    def _readout_waveform(self, t_list):
        t_start = 0
        t_duration = T_READOUT / 1.5
        rise_time = 50

        t0 = 0
        t1 = t0 + t_start  # 立ち上がり開始時刻
        t2 = t1 + rise_time  # 立ち上がり完了時刻
        t3 = t2 + t_duration  # 立ち下がり開始時刻
        t4 = t3 + rise_time  # 立ち下がり完了時刻

        cond_12 = (t1 <= t_list) & (t_list < t2)  # 立ち上がり時間領域の条件ブール値
        cond_23 = (t2 <= t_list) & (t_list < t3)  # 一定値領域の条件ブール値
        cond_34 = (t3 <= t_list) & (t_list < t4)  # 立ち下がり時間領域の条件ブール値

        t_12 = t_list[cond_12]  # 立ち上がり時間領域の時間リスト
        t_23 = t_list[cond_23]  # 一定値領域の時間リスト
        t_34 = t_list[cond_34]  # 立ち下がり時間領域の時間リスト

        waveform = t_list + 0 * 1j  # 波形リストの雛形
        waveform[:] = 0  # 波形リストの初期化
        waveform[cond_12] = (
            1.0 - np.cos(np.pi * (t_12 - t1) / rise_time)
        ) / 2 + 1j * 0.0  # 立ち上がり時間領域
        waveform[cond_23] = 1.0 + 1j * 0.0  # 一定値領域
        waveform[cond_34] = (
            1.0 - np.cos(np.pi * (t4 - t_34) / rise_time)
        ) / 2 + 1j * 0.0  # 立ち下がり時間領域
        # waveform[cond_34] = (1.0 + np.cos(np.pi*(t_34-t3)/rise_time)) / 2 + 1j*0.0 # 立ち下がり時間領域

        return waveform

    def _received_waveforms(
        self,
        read_qubits: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        waveforms = {}
        timestamps = {}
        for qubit in read_qubits:
            slot = self._read_rx_slot(qubit)
            waveforms[qubit] = slot.iq
            timestamps[qubit] = slot.timestamp
        return waveforms, timestamps

    def _integrated_iq(
        self,
        waveforms: dict[str, np.ndarray],
    ) -> dict[str, complex]:
        iq = {
            qubit: waveform[self.read_range].mean()
            for qubit, waveform in waveforms.items()
        }
        return iq

    def _measure(
        self,
        ctrl_qubits: list[str],
        read_qubits: list[str],
        waveforms: dict[str, np.ndarray],
    ) -> dict[str, complex]:
        self.set_circuit(
            ctrl_qubits=ctrl_qubits,
            read_qubits=read_qubits,
            waveforms=waveforms,
        )

        run(
            self.schedule,
            repeats=self.repeats,
            interval=self.interval,
            adda_to_channels=self.adda_to_channels,
            triggers=self.triggers,
        )

        rx_waveforms, _ = self._received_waveforms(read_qubits)
        result: dict[str, complex] = self._integrated_iq(rx_waveforms)
        return result

    def measure(
        self,
        waveforms: dict[str, np.ndarray],
    ) -> dict[str, complex]:
        qubits = list(waveforms.keys())

        result: dict[str, complex] = self._measure(
            ctrl_qubits=qubits,
            read_qubits=qubits,
            waveforms=waveforms,
        )
        return result

    def rabi_check(
        self,
        time_range=np.arange(0, 201, 10),
    ) -> dict[str, MeasurementResult]:
        amplitudes = ampl_hpi_dict[self.qube_id]
        result = self._rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
        )
        return result

    def rabi_experiment(
        self,
        qubit: str,
        amplitude: float,
        time_range=np.arange(0, 201, 10),
    ) -> MeasurementResult:
        result = self._rabi_experiment(
            amplitudes={qubit: amplitude},
            time_range=time_range,
        )[qubit]
        return result

    def _rabi_experiment(
        self,
        amplitudes: dict[str, float],
        time_range: np.ndarray,
    ) -> dict[str, MeasurementResult]:
        qubits = list(amplitudes.keys())

        ctrl_qubits = qubits
        read_qubits = qubits

        states: dict[str, list[complex]] = {qubit: [] for qubit in qubits}
        states_rotated: dict[str, np.ndarray] = {}
        phase_shift: dict[str, float] = {}

        for idx, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                ).values
                for qubit in qubits
            }

            ctrl_waveforms, read_waveforms = self.set_circuit(
                ctrl_qubits=ctrl_qubits,
                read_qubits=read_qubits,
                waveforms=waveforms,
            )

            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )

            rx_waveform, rx_time = self._received_waveforms(read_qubits)
            iq = self._integrated_iq(rx_waveform)

            for qubit in read_qubits:
                states[qubit].append(iq[qubit])
                states_rotated[qubit], phase_shift[qubit] = fit_and_rotate(
                    states[qubit]
                )

            clear_output(True)
            self.show_measurement_results(
                read_qubits=read_qubits,
                rx_time=rx_time,
                rx_waveform=rx_waveform,
                sweep_range=time_range[: idx + 1],
                states=states,
                states_rotated=states_rotated,
            )
            self.show_pulse_sequences(
                ctrl_qubits=ctrl_qubits,
                ctrl_waveforms=ctrl_waveforms,
                read_qubits=read_qubits,
                read_waveforms=read_waveforms,
            )
            print(f"{idx+1}/{len(time_range)}: {duration} ns")

        result = {
            qubit: MeasurementResult(
                qubit=qubit,
                sweep_range=time_range,
                data=np.array(values),
                phase_shift=phase_shift[qubit],
            )
            for qubit, values in states.items()
        }
        return result

    def repeat_pulse(
        self,
        qubit: str,
        waveform: Waveform,
        n: int,
    ) -> MeasurementResult:
        result = self.sweep_pramameter(
            qubit=qubit,
            sweep_range=np.arange(n + 1),
            waveform=lambda x: Sequence([waveform] * int(x)),
            pulse_count=1,
        )
        return result

    def sweep_pramameter(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        waveform: Callable[[float], Waveform],
        pulse_count=1,
        rabi_params: Optional[RabiParams] = None,
    ) -> MeasurementResult:
        qubits = [qubit]
        ctrl_qubits = qubits
        read_qubits = qubits

        states: dict[str, list[complex]] = {qubit: [] for qubit in qubits}
        states_rotated = {}
        phase_shift = {}

        for idx, var in enumerate(sweep_range):
            pulse = waveform(var)
            sequence = Sequence([pulse] * pulse_count)
            waveforms = {
                qubit: sequence.values,
            }

            ctrl_waveforms, read_waveforms = self.set_circuit(
                ctrl_qubits=ctrl_qubits,
                read_qubits=read_qubits,
                waveforms=waveforms,
            )

            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )

            rx_waveform, rx_time = self._received_waveforms(read_qubits)
            iq = self._integrated_iq(rx_waveform)

            for qubit in read_qubits:
                states[qubit].append(iq[qubit])
                if rabi_params is not None:
                    phase_shift[qubit] = rabi_params.phase_shift
                    states_rotated[qubit] = rotate(states[qubit], phase_shift[qubit])
                else:
                    states_rotated[qubit], phase_shift[qubit] = fit_and_rotate(
                        states[qubit]
                    )

            clear_output(True)
            self.show_measurement_results(
                read_qubits=read_qubits,
                rx_time=rx_time,
                rx_waveform=rx_waveform,
                sweep_range=sweep_range[: idx + 1],
                states=states,
                states_rotated=states_rotated,
            )
            self.show_pulse_sequences(
                ctrl_qubits=ctrl_qubits,
                ctrl_waveforms=ctrl_waveforms,
                read_qubits=read_qubits,
                read_waveforms=read_waveforms,
            )
            print(f"{idx+1}/{len(sweep_range)}: {var}")

        result = MeasurementResult(
            qubit=qubit,
            sweep_range=sweep_range,
            data=np.array(states[qubit]),
            phase_shift=phase_shift[qubit],
        )
        return result

    def show_pulse_sequences(
        self,
        ctrl_qubits,
        ctrl_waveforms,
        read_qubits,
        read_waveforms,
    ):
        """
        Shows the pulse sequences.
        """

        # number of qubits
        N = len(ctrl_qubits)

        # initialize the figure
        plt.figure(figsize=(15, 1.5 * (N + 1)))
        plt.subplots_adjust(hspace=0.0)  # no space between subplots

        # the grid is adjusted to the number of qubits (N)
        # the last row (N+1) is for the readout waveform
        gs = gridspec.GridSpec(N + 1, 1)

        # the x-axis range
        xlim = (-self.schedule.offset * 1e-3, T_READOUT * 1e-3)

        # initialize the axes
        # share the x-axis with the first subplot
        axes: list = []
        for i in range(N):
            if i == 0:
                ax = plt.subplot(gs[i])
                ax.set_title("Pulse waveform")
                ax.set_xlim(xlim)  # us
                ax.xaxis.set_visible(False)
                axes.append(ax)
            else:
                ax = plt.subplot(gs[i], sharex=axes[0])
                ax.xaxis.set_visible(False)
                axes.append(ax)
        ro_ax = plt.subplot(gs[N], sharex=axes[0])
        ro_ax.set_xlabel("Time / us")
        ro_ax.xaxis.set_visible(True)
        axes.append(ro_ax)

        # the list of the maximum amplitudes used for the ylim
        max_ampl_list = []

        # plot the control pulses
        # the real and imaginary parts are plotted in the same subplot
        for i, qubit in enumerate(ctrl_qubits):
            ctrl_time = self._ctrl_times(qubit) * 1e-3  # ns -> us
            ctrl_waveform = ctrl_waveforms[qubit]
            axes[i].plot(
                ctrl_time,
                np.real(ctrl_waveform),
                label=qubit + " control (real)",
            )
            axes[i].plot(
                ctrl_time,
                np.imag(ctrl_waveform),
                label=qubit + " control (imag)",
            )
            axes[i].legend()
            max_ampl_list.append(np.max(np.abs(ctrl_waveform)))

        # plot the readout pulses
        for i, qubit in enumerate(read_qubits):
            read_time = self._read_times(qubit) * 1e-3  # ns -> us
            read_waveform = read_waveforms[qubit]
            axes[N].plot(
                read_time,
                np.abs(read_waveform),
                label=qubit + " readout (abs)",
                linestyle="dashed",
            )
            axes[N].legend()
            max_ampl_list.append(np.max(np.abs(read_waveform)))

        # set the y-axis range according to the maximum amplitude
        max_ampl = np.max(max_ampl_list)
        for i in range(N + 1):
            axes[i].set_ylim(-1.1 * max_ampl, 1.1 * max_ampl)

        plt.show()

    def show_measurement_results(
        self,
        read_qubits,
        rx_time,
        rx_waveform,
        sweep_range,
        states,
        states_rotated,
    ):
        plt.figure(figsize=(15, 6 * len(read_qubits)))
        gs = gridspec.GridSpec(2 * len(read_qubits), 2, wspace=0.3, hspace=0.5)

        ax = {}
        for i, qubit in enumerate(read_qubits):
            ax[qubit] = [
                plt.subplot(gs[i * 2, 0]),
                plt.subplot(gs[i * 2 + 1, 0]),
                plt.subplot(gs[i * 2 : i * 2 + 2, 1]),
            ]

        for qubit in read_qubits:
            """検波した読み出しパルス波形表示"""
            avg_num = 50  # 平均化する個数

            mov_avg_readout_iq = (
                np.convolve(rx_waveform[qubit], np.ones(avg_num), mode="valid")
                / avg_num
            )  # 移動平均
            mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

            ax[qubit][0].plot(
                rx_time[qubit] * 1e-3,
                np.real(mov_avg_readout_iq),
                label="I",
            )
            ax[qubit][0].plot(
                rx_time[qubit] * 1e-3,
                np.imag(mov_avg_readout_iq),
                label="Q",
            )

            ax[qubit][0].plot(
                rx_time[qubit][self.read_range] * 1e-3,
                np.real(mov_avg_readout_iq)[self.read_range],
                lw=5,
            )
            ax[qubit][0].plot(
                rx_time[qubit][self.read_range] * 1e-3,
                np.imag(mov_avg_readout_iq)[self.read_range],
                lw=5,
            )

            ax[qubit][0].set_xlabel("Time / us")
            ax[qubit][0].set_xlim(0, 2.0)
            ax[qubit][0].set_title("Detected readout pulse waveform " + qubit)
            ax[qubit][0].legend()

            """Rabi振動"""
            ax[qubit][1].plot(sweep_range, np.real(states[qubit]), "o-", label="I")
            ax[qubit][1].plot(sweep_range, np.imag(states[qubit]), "o-", label="Q")
            ax[qubit][1].set_xlabel("Sweep index")
            ax[qubit][1].set_title("Detected signal " + qubit)
            ax[qubit][1].legend()

            """IQ平面上での複素振幅"""
            ax[qubit][2].plot(
                np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2
            )

            width = max(np.abs(states[qubit]))
            ax[qubit][2].set_xlim(-width, width)
            ax[qubit][2].set_ylim(-width, width)
            ax[qubit][2].plot(
                np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
            )
            ax[qubit][2].plot(
                np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
            )
            ax[qubit][2].set_xlabel("I")
            ax[qubit][2].set_ylabel("Q")
            ax[qubit][2].set_title("Complex amplitude on IQ plane " + qubit)

            ax[qubit][2].scatter(
                np.real(states[qubit]),
                np.imag(states[qubit]),
                label="Before rotation",
            )
            ax[qubit][2].scatter(
                np.real(states[qubit])[0],
                np.imag(states[qubit])[0],
                color="blue",
            )

            ax[qubit][2].scatter(
                np.real(states_rotated[qubit]),
                np.imag(states_rotated[qubit]),
                label="After rotation",
            )
            ax[qubit][2].scatter(
                np.real(states_rotated[qubit][0]),
                np.imag(states_rotated[qubit][0]),
                color="red",
            )
            ax[qubit][2].legend()
        plt.show()

    def normalize_rabi(
        self,
        result: MeasurementResult,
        wave_count=2.5,
    ):
        """
        Normalize the Rabi oscillation data.
        """
        time = result.sweep_range

        # Rotate the data to the vertical (Q) axis
        points, angle = fit_and_rotate(data=result.data)
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
        self.rabi_params[result.qubit] = rabi_params

        values_normalized = (values - rabi_params.offset) / rabi_params.amplitude
        return values_normalized, rabi_params

    def expectation_value(
        self,
        qubit: str,
        iq: complex,
    ):
        params = self.rabi_params[qubit]
        iq = iq * np.exp(-1j * params.phase_shift)
        value = iq.imag
        value = (value - params.offset) / params.amplitude
        return value
