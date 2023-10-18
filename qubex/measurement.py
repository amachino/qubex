from typing import Callable
from attr import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.typing import NDArray
from IPython.display import clear_output

import qubecalib as qc
from qubecalib.pulse import Read, Schedule, Blank, Arbit
from qubecalib.setupqube import run

qc.ui.MATPLOTLIB_PYPLOT = plt

from .pulse import Rect, Waveform, Sequence
from .analysis import rotate_to_vertical

# 実験パラメータ
from .params import ro_ampl_dict, ro_freq_dict, ctrl_freq_dict

T_READ = 128 * 12  # [ns] 128の倍数（2022/08/18）
T_CTRL = 10 * 2048  # [ns]
T_MARGIN = 128 * 2  # [ns]
READ_SLICE_RANGE = slice(200, int(200 + 800 / 1.5))  # 1000) # 読出しデータ切り取り範囲

SAMPLING_PERIOD = 2

CTRL = "CTRL_"
READ_TX = "READ_TX_"
READ_RX = "READ_RX_"

MUX = [
    ["Q00", "Q01", "Q02", "Q03"],
    ["Q04", "Q05", "Q06", "Q07"],
    ["Q08", "Q09", "Q10", "Q11"],
    ["Q12", "Q13", "Q14", "Q15"],
]


@dataclass
class MeasurementResult:
    qubit: str
    time: NDArray[np.int64]
    data: NDArray[np.complex128]

    @property
    def real(self):
        return np.real(self.data)

    @property
    def imag(self):
        return np.imag(self.data)

    @property
    def vector(self):
        return np.column_stack([self.real, self.imag])


class Measurement:
    def __init__(
        self,
        qube_id: str,
        mux_number: int,
        readout_ports: tuple[str, str] = ("port0", "port1"),
        time_ctrl=T_CTRL,
        repeats=10_000,
        interval=100_000,
    ):
        self.qube_id = qube_id
        self.qube = qc.ui.QubeControl(f"{qube_id}.yml").qube
        self.qubits = MUX[mux_number]
        self.readout_ports = readout_ports
        self.time_ctrl = time_ctrl
        self.repeats = repeats
        self.interval = interval
        self._setup(self.qube)

    def _setup(self, qube):
        port_tx = qube.ports[self.readout_ports[0]]
        port_rx = qube.ports[self.readout_ports[1]]

        port_tx.lo.mhz = 12500
        port_tx.nco.mhz = 2500 - 125
        port_tx.mix.ssb = qc.qube.SSB.LSB
        port_tx.awg0.nco.mhz = 0
        port_tx.mix.vatt = 0x800

        port_rx.nco.mhz = qube.port0.nco.mhz
        port_rx.adc.capt0.ssb = qc.qube.SSB.LSB
        port_rx.delay = 128 + 6 * 128  # [ns]

        qube.port5.lo.mhz = 10000
        qube.port5.nco.mhz = 2000 + 375
        qube.port5.awg0.nco.mhz = 0
        qube.port5.awg1.nco.mhz = 0
        qube.port5.awg2.nco.mhz = 0
        qube.port5.mix.vatt = 0x800

        qube.port6.lo.mhz = 10000
        qube.port6.nco.mhz = 2000 - 375
        qube.port6.awg0.nco.mhz = 0
        qube.port6.awg1.nco.mhz = 0
        qube.port6.awg2.nco.mhz = 0
        qube.port6.mix.vatt = 0x800

        qube.port7.lo.mhz = 10000
        qube.port7.nco.mhz = 2000 - 375
        qube.port7.awg0.nco.mhz = 0
        qube.port7.awg1.nco.mhz = 0
        qube.port7.awg2.nco.mhz = 0
        qube.port7.mix.vatt = 0x800

        qube.port8.lo.mhz = 9000
        qube.port8.nco.mhz = 2000 - 250
        qube.port8.awg0.nco.mhz = 0
        qube.port8.awg1.nco.mhz = 0
        qube.port8.awg2.nco.mhz = 0
        qube.port8.mix.vatt = 0x800

        self.all_read_qubits = self.qubits

        self.all_ctrl_qubits = [
            self.qubits[0] + "_lo",
            self.qubits[0],
            self.qubits[0] + "_hi",
            self.qubits[1] + "_lo",
            self.qubits[1],
            self.qubits[1] + "_hi",
            self.qubits[2] + "_lo",
            self.qubits[2],
            self.qubits[2] + "_hi",
            self.qubits[3] + "_lo",
            self.qubits[3],
            self.qubits[3] + "_hi",
        ]

        """Scheduleのchannel割り当て"""
        self.schedule = Schedule()

        for qubit in self.all_read_qubits:
            self.schedule.add_channel(
                key=READ_TX + qubit,
                center_frequency=ro_freq_dict[self.qube_id][qubit],
            )
            self.schedule.add_channel(
                key=READ_RX + qubit,
                center_frequency=ro_freq_dict[self.qube_id][qubit],
            )
        for qubit in self.all_ctrl_qubits:
            self.schedule.add_channel(
                key=qubit,
                center_frequency=ctrl_freq_dict[qubit],
            )

        """パルスシーケンスの時間ブロック割り当て"""
        t_read = T_READ
        t_ctrl = self.time_ctrl
        t_margin = T_MARGIN

        for qubit in self.all_read_qubits:
            (
                self.schedule[READ_TX + qubit]
                << Blank(duration=t_ctrl)
                << Arbit(duration=t_read, amplitude=1)
                << Blank(duration=4 * t_margin)
            )  # type: ignore
            (
                self.schedule[READ_RX + qubit]
                << Blank(duration=t_ctrl - t_margin)
                << Read(duration=t_read + 5 * t_margin)
            )  # type: ignore
        for qubit in self.all_ctrl_qubits:
            (
                self.schedule[qubit]
                << Arbit(duration=t_ctrl, amplitude=1)
                << Blank(duration=t_read + 4 * t_margin)
            )  # type: ignore

        durations = [v.duration for k, v in self.schedule.items()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

        self.adda_to_channels = {
            port_tx.dac.awg0: [
                self.schedule[READ_TX + qubit_] for qubit_ in self.all_read_qubits
            ],
            port_rx.adc.capt0: [
                self.schedule[READ_RX + qubit_] for qubit_ in self.all_read_qubits
            ],
            qube.port5.dac.awg0: [self.schedule[self.qubits[0] + "_lo"]],
            qube.port6.dac.awg0: [self.schedule[self.qubits[1] + "_lo"]],
            qube.port7.dac.awg0: [self.schedule[self.qubits[2] + "_lo"]],
            qube.port8.dac.awg0: [self.schedule[self.qubits[3] + "_lo"]],
            qube.port5.dac.awg1: [self.schedule[self.qubits[0]]],
            qube.port6.dac.awg1: [self.schedule[self.qubits[1]]],
            qube.port7.dac.awg1: [self.schedule[self.qubits[2]]],
            qube.port8.dac.awg1: [self.schedule[self.qubits[3]]],
            qube.port5.dac.awg2: [self.schedule[self.qubits[0] + "_hi"]],
            qube.port6.dac.awg2: [self.schedule[self.qubits[1] + "_hi"]],
            qube.port7.dac.awg2: [self.schedule[self.qubits[2] + "_hi"]],
            qube.port8.dac.awg2: [self.schedule[self.qubits[3] + "_hi"]],
        }

        self.triggers = [port_tx.dac.awg0]

        self.schedule.offset = t_ctrl  # [ns] 読み出し開始時刻（時間基準点）の設定
        self.ctrl_channels = {}
        self.ctrl_times = {}
        for qubit in self.all_ctrl_qubits:
            self.ctrl_channels[qubit] = self.schedule[qubit].findall(Arbit)[
                0
            ]  # findall(Arbit)はSchedule内からArbitの要素だけ抜き出してリスト化する
            self.ctrl_times[qubit] = (
                self.schedule[qubit].get_timestamp(self.ctrl_channels[qubit])
                - self.schedule.offset
            )  # [ns] 任意波形の時間座標の指定

        self.read_channels = {}
        self.read_times = {}
        for qubit in self.all_read_qubits:
            self.read_channels[qubit] = self.schedule[READ_TX + qubit].findall(Arbit)[0]
            self.read_times[qubit] = (
                self.schedule[READ_TX + qubit].get_timestamp(self.read_channels[qubit])
                - self.schedule.offset
            )

        self.slice_range = READ_SLICE_RANGE

    def initialize_circuit(
        self,
        ctrl_qubits: list[str],
        read_qubits: list[str],
    ):
        for qubit in ctrl_qubits:
            self.ctrl_channels[qubit].iq[:] = 0  # パルス波形の初期化

        for qubit in read_qubits:
            self.read_channels[qubit].iq[:] = 0

    def finalize_circuit(
        self,
        ctrl_qubits: list[str],
        read_qubits: list[str],
        waveforms: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        ctrl_waveforms = {}
        for qubit in ctrl_qubits:
            t_wave = SAMPLING_PERIOD * len(waveforms[qubit])  # ns
            self.ctrl_channels[qubit].iq[
                (-t_wave <= self.ctrl_times[qubit]) & (self.ctrl_times[qubit] < 0)
            ] = waveforms[
                qubit
            ]  # Arbitのパルス波形をここで指定
            ctrl_waveforms[qubit] = self.ctrl_channels[qubit].iq

        read_waveforms = {}
        for qubit in read_qubits:
            self.read_channels[qubit].iq[:] = ro_ampl_dict[self.qube_id][
                qubit
            ] * self.readout_waveform(
                self.read_times[qubit]
            )  # 読み出し波形の指定
            read_waveforms[qubit] = self.read_channels[qubit].iq

        return ctrl_waveforms, read_waveforms

    def readout_waveform(self, t_list):
        t_start = 0
        t_duration = T_READ / 1.5
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

    def get_rx_waveforms(
        self,
        read_qubits: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        waveforms = {}
        timestamps = {}
        for qubit in read_qubits:
            waveform = self.schedule[READ_RX + qubit].findall(Read)[0]
            waveforms[qubit] = waveform.iq
            timestamps[qubit] = waveform.timestamp
        return waveforms, timestamps

    def get_integrated_iq(
        self,
        waveforms: dict[str, np.ndarray],
    ) -> dict[str, complex]:
        iq = {
            qubit: waveform[self.slice_range].mean()
            for qubit, waveform in waveforms.items()
        }
        return iq

    def _measure(
        self,
        ctrl_qubits: list[str],
        read_qubits: list[str],
        waveforms: dict[str, np.ndarray],
    ) -> dict[str, complex]:
        self.initialize_circuit(
            ctrl_qubits=ctrl_qubits,
            read_qubits=read_qubits,
        )

        self.finalize_circuit(
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

        rx_waveforms, _ = self.get_rx_waveforms(read_qubits)
        result: dict[str, complex] = self.get_integrated_iq(rx_waveforms)
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
        sweep_range=np.arange(0, 200, 10),
        amplitude=0.03,
    ) -> dict[str, MeasurementResult]:
        result = self._rabi_experiment(
            time_range=sweep_range,
            amplitudes={qubit: amplitude for qubit in self.all_read_qubits},
        )
        return result

    def rabi_experiment(
        self,
        qubit: str,
        time_range: NDArray[np.int64],
        amplitude: float,
    ) -> MeasurementResult:
        result = self._rabi_experiment(
            time_range=time_range,
            amplitudes={qubit: amplitude},
        )[qubit]
        return result

    def _rabi_experiment(
        self,
        time_range: NDArray[np.int64],
        amplitudes: dict[str, float],
    ) -> dict[str, MeasurementResult]:
        qubits = list(amplitudes.keys())

        ctrl_qubits = qubits
        read_qubits = qubits

        states = {qubit: [] for qubit in qubits}
        states_rotated = {}

        for idx, duration in enumerate(time_range):
            self.initialize_circuit(
                ctrl_qubits=ctrl_qubits,
                read_qubits=read_qubits,
            )

            waveforms = {
                qubit: Rect(
                    duration,
                    amplitudes[qubit],
                ).values
                for qubit in qubits
            }

            ctrl_waveforms, read_waveforms = self.finalize_circuit(
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

            rx_waveform, rx_time = self.get_rx_waveforms(read_qubits)
            iq = self.get_integrated_iq(rx_waveform)

            for qubit in read_qubits:
                states[qubit].append(iq[qubit])
                states_rotated[qubit] = rotate_to_vertical(states[qubit])

            clear_output(True)
            self.show_measurement_results(
                read_qubits,
                rx_time,
                rx_waveform,
                time_range[: idx + 1],
                states,
                states_rotated,
            )
            self.show_pulse_sequences(
                qubits,
                ctrl_waveforms,
                ctrl_qubits,
                read_waveforms,
            )
            print(f"{idx+1}/{len(time_range)}: {duration} ns")

        result = {
            qubit: MeasurementResult(
                qubit=qubit,
                time=time_range,
                data=np.array(state),
            )
            for qubit, state in states.items()
        }
        return result

    def sweep_pramameter(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        waveform: Callable[[float], Waveform],
        pulse_count=1,
    ):
        qubits = [qubit]
        ctrl_qubits = qubits
        read_qubits = qubits

        states: dict[str, list[complex]] = {qubit: [] for qubit in qubits}
        states_rotated = {}

        for idx, var in enumerate(sweep_range):
            self.initialize_circuit(
                ctrl_qubits=ctrl_qubits,
                read_qubits=read_qubits,
            )

            pulse = waveform(var)
            sequence = Sequence([pulse] * pulse_count)
            waveforms = {
                qubit: sequence.values,
            }

            ctrl_waveforms, read_waveforms = self.finalize_circuit(
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

            rx_waveform, rx_time = self.get_rx_waveforms(read_qubits)
            iq = self.get_integrated_iq(rx_waveform)

            for qubit in read_qubits:
                states[qubit].append(iq[qubit])
                states_rotated[qubit] = rotate_to_vertical(states[qubit])

            clear_output(True)
            self.show_measurement_results(
                read_qubits,
                rx_time,
                rx_waveform,
                sweep_range[: idx + 1],
                states,
                states_rotated,
            )
            self.show_pulse_sequences(
                ctrl_qubits,
                ctrl_waveforms,
                read_qubits,
                read_waveforms,
            )
            print(f"{idx+1}/{len(sweep_range)}: {var}")

        return states[qubit]

    def repeat_pulse(
        self,
        qubit: str,
        n: int,
        waveform: Waveform,
    ):
        qubits = [qubit]
        ctrl_qubits = qubits
        read_qubits = qubits

        states: dict[str, list[complex]] = {qubit: [] for qubit in qubits}
        states_rotated = {}

        sweep_range = np.arange(n)
        for idx in sweep_range:
            self.initialize_circuit(
                ctrl_qubits=ctrl_qubits,
                read_qubits=read_qubits,
            )

            sequence = Sequence([waveform] * idx)
            waveforms = {
                qubit: sequence.values,
            }

            ctrl_waveforms, read_waveforms = self.finalize_circuit(
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

            rx_waveform, rx_time = self.get_rx_waveforms(read_qubits)
            iq = self.get_integrated_iq(rx_waveform)

            for qubit in read_qubits:
                states[qubit].append(iq[qubit])
                states_rotated[qubit] = rotate_to_vertical(states[qubit])

            clear_output(True)
            self.show_measurement_results(
                read_qubits,
                rx_time,
                rx_waveform,
                sweep_range[: idx + 1],
                states,
                states_rotated,
            )
            self.show_pulse_sequences(
                ctrl_qubits,
                ctrl_waveforms,
                read_qubits,
                read_waveforms,
            )
            print(f"{idx+1}/{len(sweep_range)}")

        return states[qubit]

    def show_pulse_sequences(
        self,
        ctrl_qubits,
        ctrl_waveforms,
        read_qubits,
        read_waveforms,
        xlim=(-3.0, 1.5),
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
            ctrl_time = self.ctrl_times[qubit] * 1e-3  # ns -> us
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
            read_time = self.read_times[qubit] * 1e-3  # ns -> us
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

            slice_range = self.slice_range
            ax[qubit][0].plot(
                rx_time[qubit][slice_range] * 1e-3,
                np.real(mov_avg_readout_iq)[slice_range],
                lw=5,
            )
            ax[qubit][0].plot(
                rx_time[qubit][slice_range] * 1e-3,
                np.imag(mov_avg_readout_iq)[slice_range],
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
                color="red",
            )

            ax[qubit][2].scatter(
                np.real(states_rotated[qubit]),
                np.imag(states_rotated[qubit]),
                label="After rotation",
            )
            ax[qubit][2].scatter(
                np.real(states_rotated[qubit][0]),
                np.imag(states_rotated[qubit][0]),
                color="blue",
            )
            ax[qubit][2].legend()
        plt.show()
