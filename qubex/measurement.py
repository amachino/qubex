from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import qubecalib as qc
from qubecalib.pulse import Read, Schedule, Blank, Arbit
from qubecalib.setupqube import run

qc.ui.MATPLOTLIB_PYPLOT = plt

from .pulse import Rect, Waveform, Sequence

from .utils import (
    raised_cos,
    show_pulse_sequences,
    show_measurement_results,
    linear_fit_and_rotate_IQ,
)

# 実験パラメータ
from .params import ro_ampl_dict, ro_freq_dict, ctrl_freq_dict

T_READ = 128 * 12  # [ns] 128の倍数（2022/08/18）
T_CTRL = 10 * 2048  # [ns]
T_MARGIN = 128 * 2  # [ns]
READ_SLICE_RANGE = slice(200, int(200 + 800 / 1.5))  # 1000) # 読出しデータ切り取り範囲

MUX = [
    ["Q00", "Q01", "Q02", "Q03"],
    ["Q04", "Q05", "Q06", "Q07"],
    ["Q08", "Q09", "Q10", "Q11"],
    ["Q12", "Q13", "Q14", "Q15"],
]


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

        self.all_ro_qubit_list = self.qubits

        self.all_ctrl_qubit_list = [
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

        for qubit_ in self.all_ro_qubit_list:
            self.schedule.add_channel(
                key="RO_send_" + qubit_,
                center_frequency=ro_freq_dict[self.qube_id][qubit_],
            )
            self.schedule.add_channel(
                key="RO_return_" + qubit_,
                center_frequency=ro_freq_dict[self.qube_id][qubit_],
            )
        for qubit_ in self.all_ctrl_qubit_list:
            self.schedule.add_channel(
                key=qubit_,
                center_frequency=ctrl_freq_dict[qubit_],
            )

        """パルスシーケンスの時間ブロック割り当て"""
        time_ro = T_READ
        time_ctrl = self.time_ctrl
        time_margin = T_MARGIN

        for qubit_ in self.all_ro_qubit_list:
            (
                self.schedule["RO_send_" + qubit_]
                << Blank(duration=time_ctrl)
                << Arbit(duration=time_ro, amplitude=1)
                << Blank(duration=4 * time_margin)
            )  # type: ignore
            (
                self.schedule["RO_return_" + qubit_]
                << Blank(duration=time_ctrl - time_margin)
                << Read(duration=time_ro + 5 * time_margin)
            )  # type: ignore
        for qubit_ in self.all_ctrl_qubit_list:
            (
                self.schedule[qubit_]
                << Arbit(duration=time_ctrl, amplitude=1)
                << Blank(duration=time_ro + 4 * time_margin)
            )  # type: ignore

        durations = [v.duration for k, v in self.schedule.items()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

        self.adda_to_channels = {
            port_tx.dac.awg0: [
                self.schedule["RO_send_" + qubit_] for qubit_ in self.all_ro_qubit_list
            ],
            port_rx.adc.capt0: [
                self.schedule["RO_return_" + qubit_]
                for qubit_ in self.all_ro_qubit_list
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

        self.schedule.offset = time_ctrl  # [ns] 読み出し開始時刻（時間基準点）の設定
        self.c = {}
        self.t_c = {}
        for qubit_ in self.all_ctrl_qubit_list:
            self.c[qubit_] = self.schedule[qubit_].findall(Arbit)[
                0
            ]  # findall(Arbit)はSchedule内からArbitの要素だけ抜き出してリスト化する
            self.t_c[qubit_] = (
                self.schedule[qubit_].get_timestamp(self.c[qubit_])
                - self.schedule.offset
            )  # [ns] 任意波形の時間座標の指定

        self.ro = {}
        self.t_ro = {}
        for qubit_ in self.all_ro_qubit_list:
            self.ro[qubit_] = self.schedule["RO_send_" + qubit_].findall(Arbit)[0]
            self.t_ro[qubit_] = (
                self.schedule["RO_send_" + qubit_].get_timestamp(self.ro[qubit_])
                - self.schedule.offset
            )

        self.slice_range = READ_SLICE_RANGE

    def finalize_circuit(self, ctrl_qubit_list_, ro_qubit_list_, waveforms_dict):
        """
        波形リストの辞書を受けて, QuBEの制御IQ波形オブジェクトを生成する.

        Parameters
        ----------
        ctrl_qubit_list_ : qubit制御チャンネルのリスト
        ro_qubit_list_ : 読み出しチャンネルのリスト
        waveforms_dict : qubit制御波形リストの辞書

        Returns
        -------
        c_iq : 制御パルスIQ波形リストの辞書
        ro_iq : 読み出しパルスIQ波形リストの辞書
        """
        c_iq = {}
        for qubit_ in ctrl_qubit_list_:
            t_wf = 2 * len(waveforms_dict[qubit_])  # ns
            self.c[qubit_].iq[
                (-t_wf <= self.t_c[qubit_]) & (self.t_c[qubit_] < 0)
            ] = waveforms_dict[
                qubit_
            ]  # Arbitのパルス波形をここで指定
            c_iq[qubit_] = self.c[qubit_].iq  # 制御パルスIQ波形リストの辞書

        ro_iq = {}
        for qubit_ in ro_qubit_list_:
            self.ro[qubit_].iq[:] = ro_ampl_dict[self.qube_id][qubit_] * raised_cos(
                self.t_ro[qubit_], 0, T_READ / 1.5, 50
            )  # 読み出し波形の指定
            ro_iq[qubit_] = self.ro[qubit_].iq

        return c_iq, ro_iq

    def initialize_circuit(self):
        for qubit_ in self.all_ctrl_qubit_list:
            self.c[qubit_].iq[:] = 0  # パルス波形の初期化

        for qubit_ in self.all_ro_qubit_list:
            self.ro[qubit_].iq[:] = 0

    def detect_ro_waveform(self, ro_qubit_list_):
        """
        Parameters
        ----------
        ro_qubit_list_ : 読み出しで用いるqubitリスト

        Returns
        -------
        IQ_sig_list_dict : 新たにIQ値を追加されたIQ_sig_list_dict
        detected_iq : 検波された読み出しIQ波形リストの辞書
        detected_time : detected_iqに対応する時間リストの辞書
        c_iq : 制御パルスIQ波形リストの辞書
        """
        detected_obj = {}
        detected_iq = {}
        detected_time = {}

        for qubit_ in ro_qubit_list_:
            detected_obj[qubit_] = self.schedule["RO_return_" + qubit_].findall(Read)[
                0
            ]  # 検出した読み出しパルスのオブジェクト
            detected_iq[qubit_] = detected_obj[qubit_].iq  # 検出した読み出しパルスのIQ波形リスト
            detected_time[qubit_] = detected_obj[
                qubit_
            ].timestamp  # [ns] 検出した読み出しパルスの時間リスト

        return detected_iq, detected_time

    def time_integrate_IQ(self, detected_iq, ro_qubit_list_):
        """
        Parameters
        ----------
        detected_iq : 検波された読み出しIQ波形リストの辞書
        ro_qubit_list_ : 読み出しで用いるqubitリスト

        Returns
        -------
        IQ_sig_dict : 時間積分されたIQ値のqubitごとの辞書

        """
        IQ_sig_dict = {}

        for qubit_ in ro_qubit_list_:
            IQ_sig_dict[qubit_] = detected_iq[qubit_][
                self.slice_range
            ].mean()  # 1点の平均複素振幅を取得

        return IQ_sig_dict

    def measure(
        self,
        waveforms: dict[str, np.ndarray],
    ):
        qubits = list(waveforms.keys())

        self.initialize_circuit()
        self.finalize_circuit(qubits, qubits, waveforms)

        run(
            self.schedule,
            repeats=self.repeats,
            interval=self.interval,
            adda_to_channels=self.adda_to_channels,
            triggers=self.triggers,
        )

        detected_iq, _ = self.detect_ro_waveform(qubits)
        result = self.time_integrate_IQ(detected_iq, qubits)
        return result

    def rabi_experiment(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        amplitude: float,
    ):
        qubits = [qubit]

        IQ_before_list_dict = {qubit: [] for qubit in qubits}
        IQ_after_list_dict = {qubit: [] for qubit in qubits}

        for i, duration in enumerate(sweep_range):
            self.initialize_circuit()

            waveforms = {
                qubit: Rect(duration, amplitude).values,
            }

            c_iq, ro_iq = self.finalize_circuit(qubits, qubits, waveforms)
            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )
            detected_iq_dict, detected_time_dict = self.detect_ro_waveform(qubits)
            IQ_sig_dict = self.time_integrate_IQ(detected_iq_dict, qubits)
            IQ_before_list_dict[qubit].append(IQ_sig_dict[qubit])
            IQ_after_list_dict, grad_dict, intercept_dict = linear_fit_and_rotate_IQ(
                qubits, IQ_before_list_dict
            )
            clear_output(True)
            show_measurement_results(
                qubits,
                detected_time_dict,
                detected_iq_dict,
                self.slice_range,
                sweep_range[: i + 1],
                IQ_before_list_dict,
                IQ_after_list_dict,
            )
            show_pulse_sequences([qubit], self.t_c, c_iq, [qubit], self.t_ro, ro_iq)
            print(duration)

        result = np.array(IQ_before_list_dict[qubit])
        return result

    def sweep_pramameter(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        waveform: Callable[[float], Waveform],
        pulse_count=1,
    ):
        qubits = [qubit]

        IQ_before_list_dict = {qubit: [] for qubit in qubits}
        IQ_after_list_dict = {qubit: [] for qubit in qubits}

        for i, v in enumerate(sweep_range):
            self.initialize_circuit()

            pulse = waveform(v)
            sequence = Sequence([pulse] * pulse_count)
            waveforms = {
                qubit: sequence.values,
            }

            c_iq, ro_iq = self.finalize_circuit(qubits, qubits, waveforms)
            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )
            detected_iq_dict, detected_time_dict = self.detect_ro_waveform(qubits)
            IQ_sig_dict = self.time_integrate_IQ(detected_iq_dict, qubits)
            IQ_before_list_dict[qubit].append(IQ_sig_dict[qubit])
            IQ_after_list_dict, grad_dict, intercept_dict = linear_fit_and_rotate_IQ(
                qubits, IQ_before_list_dict
            )
            clear_output(True)
            show_measurement_results(
                qubits,
                detected_time_dict,
                detected_iq_dict,
                self.slice_range,
                sweep_range[: i + 1],
                IQ_before_list_dict,
                IQ_after_list_dict,
            )
            show_pulse_sequences([qubit], self.t_c, c_iq, [qubit], self.t_ro, ro_iq)
            print(f"{i+1}/{len(sweep_range)}: {v}")

        result = np.array(IQ_before_list_dict[qubit])
        return result

    def sweep_pulse_duration(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        amplitude: float,
        waveform: Callable[[int, float], Waveform],
    ):
        qubits = [qubit]

        IQ_before_list_dict = {qubit: [] for qubit in qubits}
        IQ_after_list_dict = {qubit: [] for qubit in qubits}

        for i, duration in enumerate(sweep_range):
            self.initialize_circuit()

            waveforms = {
                qubit: waveform(duration, amplitude).values,
            }

            c_iq, ro_iq = self.finalize_circuit(qubits, qubits, waveforms)
            run(
                self.schedule,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )
            detected_iq_dict, detected_time_dict = self.detect_ro_waveform(qubits)
            IQ_sig_dict = self.time_integrate_IQ(detected_iq_dict, qubits)
            IQ_before_list_dict[qubit].append(IQ_sig_dict[qubit])
            IQ_after_list_dict, grad_dict, intercept_dict = linear_fit_and_rotate_IQ(
                qubits, IQ_before_list_dict
            )
            clear_output(True)
            show_measurement_results(
                qubits,
                detected_time_dict,
                detected_iq_dict,
                self.slice_range,
                sweep_range[: i + 1],
                IQ_before_list_dict,
                IQ_after_list_dict,
            )
            show_pulse_sequences([qubit], self.t_c, c_iq, [qubit], self.t_ro, ro_iq)
            print(duration)

        result = np.array(IQ_before_list_dict[qubit])
        return result

    def sweep_pulse_amplitude(
        self,
        qubit: str,
        sweep_range: np.ndarray,
        duration: int,
        waveform: Callable[[int, float], Waveform],
        pulse_count=4,
    ):
        qubits = [qubit]

        IQ_before_list_dict = {qubit: [] for qubit in qubits}
        IQ_after_list_dict = {qubit: [] for qubit in qubits}

        for i, amplitude in enumerate(sweep_range):
            self.initialize_circuit()

            pulse = waveform(duration, amplitude)
            sequence = Sequence([pulse] * pulse_count)
            waveforms = {
                qubit: sequence.values,
            }

            c_iq, ro_iq = self.finalize_circuit(qubits, qubits, waveforms)
            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )
            detected_iq_dict, detected_time_dict = self.detect_ro_waveform(qubits)
            IQ_sig_dict = self.time_integrate_IQ(detected_iq_dict, qubits)
            IQ_before_list_dict[qubit].append(IQ_sig_dict[qubit])
            IQ_after_list_dict, grad_dict, intercept_dict = linear_fit_and_rotate_IQ(
                qubits, IQ_before_list_dict
            )
            clear_output(True)
            show_measurement_results(
                qubits,
                detected_time_dict,
                detected_iq_dict,
                self.slice_range,
                sweep_range[: i + 1],
                IQ_before_list_dict,
                IQ_after_list_dict,
            )
            show_pulse_sequences([qubit], self.t_c, c_iq, [qubit], self.t_ro, ro_iq)
            print(amplitude)

        result = np.array(IQ_before_list_dict[qubit])
        return result

    def repeat_pulse(
        self,
        qubit: str,
        n: int,
        waveform: Waveform,
    ):
        qubits = [qubit]

        IQ_before_list_dict = {qubit: [] for qubit in qubits}
        IQ_after_list_dict = {qubit: [] for qubit in qubits}

        sweep_range = np.arange(n)
        for i in sweep_range:
            self.initialize_circuit()

            sequence = Sequence([waveform] * i)
            waveforms = {
                qubit: sequence.values,
            }

            c_iq, ro_iq = self.finalize_circuit(qubits, qubits, waveforms)
            run(
                self.schedule,
                repeats=self.repeats,
                interval=self.interval,
                adda_to_channels=self.adda_to_channels,
                triggers=self.triggers,
            )
            detected_iq_dict, detected_time_dict = self.detect_ro_waveform(qubits)
            IQ_sig_dict = self.time_integrate_IQ(detected_iq_dict, qubits)
            IQ_before_list_dict[qubit].append(IQ_sig_dict[qubit])
            IQ_after_list_dict, grad_dict, intercept_dict = linear_fit_and_rotate_IQ(
                qubits, IQ_before_list_dict
            )
            clear_output(True)
            show_measurement_results(
                qubits,
                detected_time_dict,
                detected_iq_dict,
                self.slice_range,
                sweep_range[: i + 1],
                IQ_before_list_dict,
                IQ_after_list_dict,
            )
            show_pulse_sequences([qubit], self.t_c, c_iq, [qubit], self.t_ro, ro_iq)

        result = np.array(IQ_before_list_dict[qubit])
        return result
