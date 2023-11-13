from typing import Final

import matplotlib.pyplot as plt

import qubecalib as qc
from qubecalib.pulse import Schedule, Channel, Blank, Arbitrary, Read
from qubecalib.setupqube import run

qc.ui.MATPLOTLIB_PYPLOT = plt  # type: ignore

from .pulse import Rect, Waveform
from .typing import (
    QubitKey,
    QubitDict,
    IQValue,
    IQArray,
    IntArray,
    ReadoutPorts,
)
from .params import (
    ctrl_freq_dict,
    ro_freq_dict,
    ro_ampl_dict,
)
from .consts import (
    MUX,
    SAMPLING_PERIOD,
    T_READOUT,
    T_MARGIN,
    READOUT_RANGE,
)

CONTROL_HIGH = "_hi"
CONTROL_LOW = "_lo"
READOUT_TX = "TX_"
READOUT_RX = "RX_"


class QubeManager:
    def __init__(
        self,
        qube_id: str,
        mux_number: int,
        readout_ports: ReadoutPorts,
        control_window: int,
    ):
        self.qube_id: Final = qube_id
        self.qube: Final = qc.ui.QubeControl(f"{qube_id}.yml").qube
        self.qubits: Final = MUX[mux_number]
        self.readout_ports: Final = readout_ports
        self.control_window: Final = control_window
        self.schedule: Final = Schedule()
        self.max_control_duration = 0
        self._init_channels()
        self._init_ports(self.qube)

    def _ctrl_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit]

    def _read_tx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READOUT_TX + qubit]

    def _read_rx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[READOUT_RX + qubit]

    def _ctrl_channels(self) -> list[Channel]:
        return [self._ctrl_channel(qubit) for qubit in self.all_control_qubits]

    def _read_tx_channels(self) -> list[Channel]:
        return [self._read_tx_channel(qubit) for qubit in self.all_readout_qubits]

    def _read_rx_channels(self) -> list[Channel]:
        return [self._read_rx_channel(qubit) for qubit in self.all_readout_qubits]

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

    def _read_tx_times(self, qubit: QubitKey) -> IntArray:
        channel = self._read_tx_channel(qubit)
        slot = self._read_tx_slot(qubit)
        local_times = channel.get_timestamp(slot)
        global_times = local_times - self.schedule.offset
        return global_times

    def _init_channels(self):
        self.all_readout_qubits = self.qubits
        self.all_control_qubits = []
        for qubit in self.qubits:
            self.all_control_qubits.extend(
                [qubit + CONTROL_LOW, qubit, qubit + CONTROL_HIGH]
            )
        for qubit in self.all_control_qubits:
            self.schedule[qubit] = Channel(
                center_frequency=ctrl_freq_dict[qubit],
            )
        for qubit in self.all_readout_qubits:
            self.schedule[READOUT_TX + qubit] = Channel(
                center_frequency=ro_freq_dict[self.qube_id][qubit],
            )
            self.schedule[READOUT_RX + qubit] = Channel(
                center_frequency=ro_freq_dict[self.qube_id][qubit],
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
            qube.port5.dac.awg0: [self._ctrl_channel(self.qubits[0] + CONTROL_LOW)],
            qube.port6.dac.awg0: [self._ctrl_channel(self.qubits[1] + CONTROL_LOW)],
            qube.port7.dac.awg0: [self._ctrl_channel(self.qubits[2] + CONTROL_LOW)],
            qube.port8.dac.awg0: [self._ctrl_channel(self.qubits[3] + CONTROL_LOW)],
            qube.port5.dac.awg1: [self._ctrl_channel(self.qubits[0])],
            qube.port6.dac.awg1: [self._ctrl_channel(self.qubits[1])],
            qube.port7.dac.awg1: [self._ctrl_channel(self.qubits[2])],
            qube.port8.dac.awg1: [self._ctrl_channel(self.qubits[3])],
            qube.port5.dac.awg2: [self._ctrl_channel(self.qubits[0] + CONTROL_HIGH)],
            qube.port6.dac.awg2: [self._ctrl_channel(self.qubits[1] + CONTROL_HIGH)],
            qube.port7.dac.awg2: [self._ctrl_channel(self.qubits[2] + CONTROL_HIGH)],
            qube.port8.dac.awg2: [self._ctrl_channel(self.qubits[3] + CONTROL_HIGH)],
        }

    def _set_waveforms(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
    ):
        self.schedule.offset = self.control_window

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

        for qubit in control_qubits:
            values = control_waveforms[qubit].values
            T = len(values) * SAMPLING_PERIOD
            t = self._ctrl_times(qubit)
            self._ctrl_slot(qubit).iq[(-T <= t) & (t < 0)] = values

        for qubit in readout_qubits:
            ampl = ro_ampl_dict[self.qube_id][qubit]
            readout_waveform = Rect(
                duration=T_READOUT,
                amplitude=ampl,
                risetime=50,
            )
            self._read_tx_slot(qubit).iq[:] = readout_waveform.values

    def get_control_frequency(
        self,
        qubit: QubitKey,
    ) -> float:
        return self.schedule[qubit].center_frequency

    def set_control_frequency(
        self,
        qubit: QubitKey,
        frequency: float,
    ):
        self.schedule[qubit].center_frequency = frequency

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

    def get_max_control_duration(
        self,
    ) -> int:
        waveforms = self.get_control_waveforms(self.all_control_qubits)
        max_length = max([len(waveform) for waveform in waveforms.values()])
        return max_length * SAMPLING_PERIOD

    def get_readout_tx_waveforms(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IQArray]:
        return {qubit: self._read_tx_slot(qubit).iq for qubit in read_qubits}

    def get_readout_tx_times(
        self,
        read_qubits: list[QubitKey],
    ) -> QubitDict[IntArray]:
        return {qubit: self._read_tx_times(qubit) for qubit in read_qubits}

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

    def measure(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
        repeats: int = 10_000,
        interval: int = 150_000,
    ) -> QubitDict[IQValue]:
        self._set_waveforms(
            control_qubits=control_qubits,
            readout_qubits=readout_qubits,
            control_waveforms=control_waveforms,
        )
        run(
            schedule=self.schedule,
            repeats=repeats,
            interval=interval,
            adda_to_channels=self.adda_to_channels,
            triggers=self.triggers,
        )
        rx_waveforms = self.get_readout_rx_waveforms(readout_qubits)
        result = {
            qubit: waveform[READOUT_RANGE].mean()
            for qubit, waveform in rx_waveforms.items()
        }
        return result
