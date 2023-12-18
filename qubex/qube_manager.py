"""
Provides a manager class `QubeManager` for setting up and executing quantum
experiments on QuBE devices. This module facilitates the configuration, control,
and readout of qubits in a quantum computing environment.
"""

from typing import Final, Optional

import matplotlib.pyplot as plt
import qubecalib as qc
from qubecalib.pulse import Arbitrary, Blank, Channel, Read, Schedule
from qubecalib.setupqube import run
from qubecalib.qube import SSB

from .configs import Configs
from .consts import SAMPLING_PERIOD, T_CONTROL, T_MARGIN, T_READOUT
from .pulse import Rect, Waveform
from .typing import IntArray, IQArray, IQValue, QubitDict, QubitKey

READOUT_TX: Final = "_TX"
READOUT_RX: Final = "_RX"

CONTROL_HI: Final = "_hi"
CONTROL_LO: Final = "_lo"

READOUT_DELAY: Final[int] = 7 * 128  # [ns]

DEFAULT_REPEATS: Final[int] = 10_000
DEFAULT_INTERVAL: Final[int] = 150_000  # [ns]


class QubeManager:
    """
    Manages quantum experiments on QuBE devices, including setup, control, and
    readout of qubits.

    Parameters
    ----------
    configs : Configs
        Configuration settings for the QuBE device and experiment.
    readout_window : int, optional
        Duration of the readout window in nanoseconds. Defaults to T_READOUT.
    control_window : int, optional
        Duration of the control window in nanoseconds. Defaults to T_CONTROL.

    Attributes
    ----------
    configs : Final
        Configuration settings for the QuBE device and experiment.
    readout_window : Final
        Duration of the readout window in nanoseconds.
    control_window : Final
        Duration of the control window in nanoseconds.
    qubits : Final
        List of qubit identifiers.
    params : Final
        Parameters related to qubit control and readout.
    qube : Optional[qc.qube.QubeTypeA]
        Instance of the QuBE device, if connected.
    schedule : Final
        Schedule of control and readout pulses for the experiment.
    readout_range : Final
        Slice object defining the range for readout signal analysis.
    """

    def __init__(
        self,
        configs: Configs,
        readout_window: int = T_READOUT,
        control_window: int = T_CONTROL,
    ):
        self.configs: Final = configs
        self.readout_window: Final = readout_window
        self.control_window: Final = control_window
        self.qubits: Final = configs.qubits
        self.params: Final = configs.params
        self.qube: Optional[qc.qube.QubeTypeA] = None
        self.schedule: Final = Schedule()
        self.readout_range = slice(T_MARGIN // 2, (self.readout_window + T_MARGIN) // 2)
        qc.ui.MATPLOTLIB_PYPLOT = plt  # type: ignore
        self._init_channels()

    def connect(self, ui: bool = True):
        """
        Connects to the QuBE device.

        Parameters
        ----------
        ui : bool, optional
            Whether to use the QuBE UI. Defaults to True.
        """
        if ui:
            self.qube = qc.ui.QubeControl(f"{self.configs.qube_id}.yml").qube
        else:
            self.qube = qc.qube.Qube.create(f"{self.configs.qube_id}.yml")
        self._init_qube()

    def loopback_mode(self, use_loopback: bool):
        """
        Sets the QuBE to loopback mode.

        Parameters
        ----------
        use_loopback : bool
            Whether to use loopback mode.
        """
        if self.qube is None:
            raise RuntimeError("QuBE is not connected.")

        if use_loopback:
            self.qube.gpio.write_value(0xFFFF)
        else:
            self.qube.gpio.write_value(0x0000)

    def get_control_frequency(self, qubit: QubitKey) -> float:
        """Returns the control frequency of the given qubit in MHz."""
        return self._ctrl_channel(qubit).center_frequency

    def get_control_lo_frequency(self, qubit: QubitKey) -> float:
        """Returns the control (lo) frequency of the given qubit in MHz."""
        return self._ctrl_lo_channel(qubit).center_frequency

    def get_control_hi_frequency(self, qubit: QubitKey) -> float:
        """Returns the control (hi) frequency of the given qubit in MHz."""
        return self._ctrl_hi_channel(qubit).center_frequency

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control frequency of the given qubit in MHz."""
        self._ctrl_channel(qubit).center_frequency = frequency

    def set_control_lo_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control (lo) frequency of the given qubit in MHz."""
        self._ctrl_lo_channel(qubit).center_frequency = frequency

    def set_control_hi_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control (hi) frequency of the given qubit in MHz."""
        self._ctrl_hi_channel(qubit).center_frequency = frequency

    def get_control_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control waveforms of the given qubits."""
        return {qubit: self._ctrl_slots[qubit].iq for qubit in qubits}

    def get_control_lo_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control (lo) waveforms of the given qubits."""
        return {qubit: self._ctrl_lo_slots[qubit].iq for qubit in qubits}

    def get_control_hi_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control (hi) waveforms of the given qubits."""
        return {qubit: self._ctrl_hi_slots[qubit].iq for qubit in qubits}

    def get_readout_tx_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the readout (tx) waveforms of the given qubits."""
        return {qubit: self._read_tx_slots[qubit].iq for qubit in qubits}

    def get_readout_rx_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the readout (rx) waveforms of the given qubits."""
        return {qubit: self._read_rx_slots[qubit].iq for qubit in qubits}

    def get_control_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control times of the given qubits."""
        return {qubit: self._ctrl_times[qubit] for qubit in qubits}

    def get_control_lo_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control (lo) times of the given qubits."""
        return {qubit: self._ctrl_lo_times[qubit] for qubit in qubits}

    def get_control_hi_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control (hi) times of the given qubits."""
        return {qubit: self._ctrl_hi_times[qubit] for qubit in qubits}

    def get_readout_tx_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the readout (tx) times of the given qubits."""
        return {qubit: self._read_tx_times[qubit] for qubit in qubits}

    def get_readout_rx_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the readout (rx) times of the given qubits."""
        return {qubit: self._read_rx_times[qubit] for qubit in qubits}

    def measure(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
        repeats: int = DEFAULT_REPEATS,
        interval: int = DEFAULT_INTERVAL,
    ) -> QubitDict[IQValue]:
        """
        Executes a quantum measurement on the given qubits.

        Parameters
        ----------
        control_qubits : list[QubitKey]
            List of qubits to control.
        readout_qubits : list[QubitKey]
            List of qubits to readout.
        control_waveforms : QubitDict[Waveform]
            Dictionary of control waveforms for each qubit.
        repeats : int, optional
            Number of times to repeat the measurement. Defaults to DEFAULT_REPEATS.
        interval : int, optional
            Interval between repeats in nanoseconds. Defaults to DEFAULT_INTERVAL.
        """
        self._set_waveforms(
            control_qubits=control_qubits,
            readout_qubits=readout_qubits,
            control_waveforms=control_waveforms,
        )
        run(
            schedule=self.schedule,
            repeats=repeats,
            interval=interval,
            adda_to_channels=self._adda_to_channels,
            triggers=self._triggers,
        )
        rx_waveforms = self.get_readout_rx_waveforms(readout_qubits)
        result = {
            qubit: waveform[self.readout_range].mean()
            for qubit, waveform in rx_waveforms.items()
        }
        return result

    def _ctrl_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit]

    def _ctrl_lo_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + CONTROL_LO]

    def _ctrl_hi_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + CONTROL_HI]

    def _read_tx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + READOUT_TX]

    def _read_rx_channel(self, qubit: QubitKey) -> Channel:
        return self.schedule[qubit + READOUT_RX]

    def _init_channels(self):
        self._ctrl_slots = {}
        self._ctrl_lo_slots = {}
        self._ctrl_hi_slots = {}
        self._read_tx_slots = {}
        self._read_rx_slots = {}

        self._ctrl_times = {}
        self._ctrl_lo_times = {}
        self._ctrl_hi_times = {}
        self._read_tx_times = {}
        self._read_rx_times = {}

        self.schedule.offset = -self.control_window

        for qubit in self.qubits:
            self._init_control_channels(qubit)
            self._init_readout_channels(qubit)
            self._init_control_slots(qubit)
            self._init_readout_slots(qubit)

        durations = [channel.duration for channel in self.schedule.values()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

    def _init_control_channels(self, qubit: QubitKey):
        control_frequency = self.params.transmon_dressed_frequency_ge[qubit]
        anharmonicity = self.params.anharmonicity[qubit]
        # control
        self.schedule[qubit] = Channel(
            center_frequency=control_frequency,
        )
        # control (lo)
        self.schedule[qubit + CONTROL_LO] = Channel(
            center_frequency=control_frequency + anharmonicity,
        )
        # control (hi)
        self.schedule[qubit + CONTROL_HI] = Channel(
            center_frequency=control_frequency,  # tmp
        )

    def _init_readout_channels(self, qubit: QubitKey):
        readout_frequency = self.params.cavity_frequency[qubit]
        # readout (tx)
        self.schedule[qubit + READOUT_TX] = Channel(
            center_frequency=readout_frequency,
        )
        # readout (rx)
        self.schedule[qubit + READOUT_RX] = Channel(
            center_frequency=readout_frequency,
        )

    def _init_control_slots(self, qubit: QubitKey):
        # control
        ctrl_slot = Arbitrary(duration=self.control_window)
        ctrl_ch = self._ctrl_channel(qubit)
        ctrl_ch.clear()
        ctrl_ch.append(ctrl_slot)
        ctrl_ch.append(Blank(duration=self.readout_window + 4 * T_MARGIN))
        self._ctrl_slots[qubit] = ctrl_slot
        self._ctrl_times[qubit] = (
            ctrl_ch.get_timestamp(ctrl_slot) + self.schedule.offset
        )

        # control (lo)
        ctrl_lo_slot = Arbitrary(duration=self.control_window, amplitude=1)
        ctrl_lo_ch = self._ctrl_lo_channel(qubit)
        ctrl_lo_ch.clear()
        ctrl_lo_ch.append(ctrl_lo_slot)
        ctrl_lo_ch.append(Blank(duration=self.readout_window + 4 * T_MARGIN))
        self._ctrl_lo_slots[qubit] = ctrl_lo_slot
        self._ctrl_lo_times[qubit] = (
            ctrl_lo_ch.get_timestamp(ctrl_lo_slot) + self.schedule.offset
        )

        # control (hi)
        ctrl_hi_slot = Arbitrary(duration=self.control_window, amplitude=1)
        ctrl_hi_ch = self._ctrl_hi_channel(qubit)
        ctrl_hi_ch.clear()
        ctrl_hi_ch.append(ctrl_hi_slot)
        ctrl_hi_ch.append(Blank(duration=self.readout_window + 4 * T_MARGIN))
        self._ctrl_hi_slots[qubit] = ctrl_hi_slot
        self._ctrl_hi_times[qubit] = (
            ctrl_hi_ch.get_timestamp(ctrl_hi_slot) + self.schedule.offset
        )

    def _init_readout_slots(self, qubit: QubitKey):
        # readout (tx)
        read_tx_slot = Arbitrary(duration=self.readout_window, amplitude=1)
        read_tx_ch = self._read_tx_channel(qubit)
        read_tx_ch.clear()
        read_tx_ch.append(Blank(duration=self.control_window))
        read_tx_ch.append(read_tx_slot)
        read_tx_ch.append(Blank(duration=4 * T_MARGIN))
        self._read_tx_slots[qubit] = read_tx_slot
        self._read_tx_times[qubit] = (
            read_tx_ch.get_timestamp(read_tx_slot) + self.schedule.offset
        )

        # readout (rx)
        read_rx_slot = Read(duration=self.readout_window + 5 * T_MARGIN)
        read_rx_ch = self._read_rx_channel(qubit)
        read_rx_ch.clear()
        read_rx_ch.append(Blank(duration=self.control_window - T_MARGIN))
        read_rx_ch.append(read_rx_slot)
        self._read_rx_slots[qubit] = read_rx_slot
        self._read_rx_times[qubit] = read_rx_slot.timestamp

    def _init_qube(self):
        if self.qube is None:
            raise RuntimeError("QuBE is not connected.")

        ports = self.qube.ports

        tx = self.configs.readout_ports[0]
        rx = self.configs.readout_ports[1]

        config_tx = self.configs.ports[tx]
        ports[tx].lo.mhz = config_tx.lo
        ports[tx].nco.mhz = config_tx.nco
        ports[tx].mix.ssb = SSB.LSB if config_tx.ssb == "LSB" else SSB.USB
        ports[tx].awg0.nco.mhz = config_tx.awg0
        ports[tx].mix.vatt = config_tx.vatt

        ports[rx].nco.mhz = ports[tx].nco.mhz
        ports[rx].adc.capt0.ssb = ports[tx].mix.ssb
        ports[rx].delay = READOUT_DELAY

        for control_port in self.configs.control_ports:
            config = self.configs.ports[control_port]
            port = ports[control_port]
            port.lo.mhz = config.lo
            port.nco.mhz = config.nco
            port.awg0.nco.mhz = config.awg0
            port.awg1.nco.mhz = config.awg1
            port.awg2.nco.mhz = config.awg2
            port.mix.vatt = config.vatt

        self.qube.gpio.write_value(0x0000)  # loopback off

        # pylint: disable=attribute-defined-outside-init
        self._triggers = [ports[tx].dac.awg0]

        adda_to_channels = {
            ports[tx].dac.awg0: [self._read_tx_channel(qubit) for qubit in self.qubits],
            ports[rx].adc.capt0: [
                self._read_rx_channel(qubit) for qubit in self.qubits
            ],
        }
        for index, control_port in enumerate(self.configs.control_ports):
            port = ports[control_port]
            qubit = self.qubits[index]
            adda_to_channels[port.dac.awg0] = [self._ctrl_lo_channel(qubit)]
            adda_to_channels[port.dac.awg1] = [self._ctrl_channel(qubit)]
            adda_to_channels[port.dac.awg2] = [self._ctrl_hi_channel(qubit)]
        self._adda_to_channels = adda_to_channels
        # pylint: enable=attribute-defined-outside-init

    def _set_waveforms(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform],
    ):
        self._set_control_waveforms(control_qubits, control_waveforms)

        readout_amplitude = self.params.readout_amplitude
        tau = 50
        readout_waveforms = {
            qubit: Rect(
                duration=T_READOUT - tau,
                amplitude=readout_amplitude[qubit],
                tau=tau,
            )
            for qubit in readout_qubits
        }
        self._set_readout_waveforms(readout_qubits, readout_waveforms)

    def _set_control_waveforms(
        self,
        qubits: list[QubitKey],
        waveforms: QubitDict[Waveform],
    ):
        for qubit in qubits:
            values = waveforms[qubit].values
            T = len(values) * SAMPLING_PERIOD
            t = self._ctrl_times[qubit]
            self._ctrl_slots[qubit].iq[(-T <= t) & (t < 0)] = values

    def _set_readout_waveforms(
        self,
        qubits: list[QubitKey],
        waveforms: QubitDict[Waveform],
    ):
        for qubit in qubits:
            values = waveforms[qubit].values
            self._read_tx_slots[qubit].iq[:] = values
