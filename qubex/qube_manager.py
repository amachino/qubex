"""
Provides a manager class `QubeManager` for setting up and executing quantum
experiments on QuBE devices. This module facilitates the configuration, control,
and readout of qubits in a quantum computing environment.
"""

from __future__ import annotations

from typing import Final, Optional

import matplotlib.pyplot as plt
import numpy as np

# mypy: disable-error-code="import-untyped"
import qubecalib as qc
from qubecalib.pulse import Arbitrary, Blank, Channel, Read, Schedule
from qubecalib.qube import AWG, CPT, SSB
from qubecalib.setupqube import run

from .configs import Configs
from .consts import MIN_SAMPLE, SAMPLING_PERIOD, T_CONTROL, T_MARGIN, T_READOUT
from .pulse import FlatTop, Waveform
from .singleshot import singleshot
from .typing import IntArray, IQArray, IQValue, QubitDict, QubitKey

READOUT_TX: Final = "_TX"
READOUT_RX: Final = "_RX"

CONTROL_CR: Final = "_CR"
CONTROL_EF: Final = "_ef"

READOUT_DELAY: Final[int] = 7 * 128  # [ns]

DEFAULT_SHOTS: Final[int] = 1024
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
    readout_range : slice
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
        self.readout_range = slice(MIN_SAMPLE, readout_window // 2 + MIN_SAMPLE)
        self._schedule: Final = Schedule()
        self._triggers: list[AWG]
        self._adda_to_channels: dict[AWG | CPT, list[Channel]]
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
            self.qube = qc.qube.Qube.create(f"{self.configs.qube_id}.yml")  # type: ignore
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

    def get_control_ef_frequency(self, qubit: QubitKey) -> float:
        """Returns the control (ef) frequency of the given qubit in MHz."""
        return self._ctrl_ef_channel(qubit).center_frequency

    def get_control_cr_frequency(self, qubit: QubitKey) -> float:
        """Returns the control (cr) frequency of the given qubit in MHz."""
        return self._ctrl_cr_channel(qubit).center_frequency

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control frequency of the given qubit in MHz."""
        self._ctrl_channel(qubit).center_frequency = frequency

    def set_control_ef_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control (ef) frequency of the given qubit in MHz."""
        self._ctrl_ef_channel(qubit).center_frequency = frequency

    def set_control_cr_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control (cr) frequency of the given qubit in MHz."""
        self._ctrl_cr_channel(qubit).center_frequency = frequency

    def get_control_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control waveforms of the given qubits."""
        return {qubit: self._ctrl_slots[qubit].iq for qubit in qubits}

    def get_control_ef_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control (ef) waveforms of the given qubits."""
        return {qubit: self._ctrl_ef_slots[qubit].iq for qubit in qubits}

    def get_control_cr_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the control (cr) waveforms of the given qubits."""
        return {qubit: self._ctrl_cr_slots[qubit].iq for qubit in qubits}

    def get_readout_tx_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the readout (tx) waveforms of the given qubits."""
        return {qubit: self._read_tx_slots[qubit].iq for qubit in qubits}

    def get_readout_rx_waveforms(self, qubits: list[QubitKey]) -> QubitDict[IQArray]:
        """Returns the readout (rx) waveforms of the given qubits."""
        return {qubit: self._read_rx_slots[qubit].iq for qubit in qubits}

    def get_control_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control times of the given qubits."""
        return {qubit: self._ctrl_times[qubit] for qubit in qubits}

    def get_control_ef_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control (ef) times of the given qubits."""
        return {qubit: self._ctrl_ef_times[qubit] for qubit in qubits}

    def get_control_cr_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the control (cr) times of the given qubits."""
        return {qubit: self._ctrl_cr_times[qubit] for qubit in qubits}

    def get_readout_tx_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the readout (tx) times of the given qubits."""
        return {qubit: self._read_tx_times[qubit] for qubit in qubits}

    def get_readout_rx_times(self, qubits: list[QubitKey]) -> QubitDict[IntArray]:
        """Returns the readout (rx) times of the given qubits."""
        return {qubit: self._read_rx_times[qubit] for qubit in qubits}

    def singleshot(
        self,
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform | IQArray | list[complex]],
        control_frequencies: Optional[QubitDict[float]] = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> QubitDict[IQArray]:
        """
        Executes a quantum measurement on the given qubits.

        Parameters
        ----------
        readout_qubits : list[QubitKey]
            List of qubits to readout.
        control_waveforms : QubitDict[Waveform]
            Dictionary of control waveforms for each qubit.
        control_frequencies : Optional[QubitDict[float]], optional
            Dictionary of control frequencies for each qubit. Defaults to None.
        shots : int, optional
            Number of shots to repeat the measurement. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between repeats in nanoseconds. Defaults to DEFAULT_INTERVAL.
        """

        # set waveforms
        self._set_waveforms(
            control_waveforms=self._normalize_waveform(control_waveforms),
            readout_qubits=readout_qubits,
        )

        # set control frequencies
        current_frequencies = {}
        if control_frequencies is not None:
            for qubit, frequency in control_frequencies.items():
                current_frequencies[qubit] = self.get_control_frequency(qubit)
                self.set_control_frequency(qubit, frequency)

        singleshot(
            adda_to_channels=self._adda_to_channels,
            triggers=self._triggers,
            readout_range=self.readout_range,
            shots=shots,
            interval=interval,
        )

        # reset control frequencies
        if control_frequencies is not None:
            for qubit, frequency in current_frequencies.items():
                self.set_control_frequency(qubit, frequency)

        # get results
        result = self.get_readout_rx_waveforms(readout_qubits)
        return result

    def measure(
        self,
        readout_qubits: list[QubitKey],
        control_waveforms: QubitDict[Waveform | IQArray | list[complex]],
        control_frequencies: Optional[QubitDict[float]] = None,
        repeats: int = DEFAULT_REPEATS,
        interval: int = DEFAULT_INTERVAL,
    ) -> QubitDict[IQValue]:
        """
        Executes a quantum measurement on the given qubits.

        Parameters
        ----------
        readout_qubits : list[QubitKey]
            List of qubits to readout.
        control_waveforms : QubitDict[Waveform]
            Dictionary of control waveforms for each qubit.
        control_frequencies : Optional[QubitDict[float]], optional
            Dictionary of control frequencies for each qubit. Defaults to None.
        repeats : int, optional
            Number of times to repeat the measurement. Defaults to DEFAULT_REPEATS.
        interval : int, optional
            Interval between repeats in nanoseconds. Defaults to DEFAULT_INTERVAL.
        """

        # set waveforms
        self._set_waveforms(
            control_waveforms=self._normalize_waveform(control_waveforms),
            readout_qubits=readout_qubits,
        )

        # set control frequencies
        current_frequencies = {}
        if control_frequencies is not None:
            for qubit, frequency in control_frequencies.items():
                current_frequencies[qubit] = self.get_control_frequency(qubit)
                self.set_control_frequency(qubit, frequency)

        # run experiment
        run(
            schedule=self._schedule,
            repeats=repeats,
            interval=interval,
            adda_to_channels=self._adda_to_channels,
            triggers=self._triggers,
        )

        # reset control frequencies
        if control_frequencies is not None:
            for qubit, frequency in current_frequencies.items():
                self.set_control_frequency(qubit, frequency)

        # get results
        rx_waveforms = self.get_readout_rx_waveforms(readout_qubits)
        result = {
            qubit: waveform[self.readout_range].mean()
            for qubit, waveform in rx_waveforms.items()
        }
        return result

    def _normalize_waveform(
        self,
        waveforms: QubitDict[Waveform | IQArray | list[complex]],
    ) -> QubitDict[IQArray]:
        """Normalizes the given waveforms to IQArray."""
        waveform_values = {}
        for qubit, waveform in waveforms.items():
            if isinstance(waveform, Waveform):
                waveform_values[qubit] = waveform.values
            elif isinstance(waveform, list):
                waveform_values[qubit] = np.array(waveform)
            else:
                waveform_values[qubit] = waveform
        return waveform_values

    def _ctrl_channel(self, qubit: QubitKey) -> Channel:
        return self._schedule[qubit]

    def _ctrl_ef_channel(self, qubit: QubitKey) -> Channel:
        return self._schedule[qubit + CONTROL_EF]

    def _ctrl_cr_channel(self, qubit: QubitKey) -> Channel:
        return self._schedule[qubit + CONTROL_CR]

    def _read_tx_channel(self, qubit: QubitKey) -> Channel:
        return self._schedule[qubit + READOUT_TX]

    def _read_rx_channel(self, qubit: QubitKey) -> Channel:
        return self._schedule[qubit + READOUT_RX]

    def _init_channels(self):
        self._ctrl_slots = {}
        self._ctrl_ef_slots = {}
        self._ctrl_cr_slots = {}
        self._read_tx_slots = {}
        self._read_rx_slots = {}

        self._ctrl_times = {}
        self._ctrl_ef_times = {}
        self._ctrl_cr_times = {}
        self._read_tx_times = {}
        self._read_rx_times = {}

        self._schedule.offset = -self.control_window

        for qubit in self.qubits:
            self._init_control_channels(qubit)
            self._init_readout_channels(qubit)
            self._init_control_slots(qubit)
            self._init_readout_slots(qubit)

        durations = [channel.duration for channel in self._schedule.values()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

    def _init_control_channels(self, qubit: QubitKey):
        control_frequency = self.params.transmon_bare_frequency_ge[qubit]
        anharmonicity = self.params.anharmonicity[qubit]
        # control
        self._schedule[qubit] = Channel(
            center_frequency=control_frequency,
        )
        # control (ef)
        self._schedule[qubit + CONTROL_EF] = Channel(
            center_frequency=control_frequency + anharmonicity,
        )
        # control (cr)
        cr_control_index = self.qubits.index(qubit)
        cr_target_index = {0: 1, 1: 3, 3: 2, 2: 0}[cr_control_index]
        cr_target_qubit = self.qubits[cr_target_index]
        cr_frequency = self.params.transmon_bare_frequency_ge[cr_target_qubit]
        self._schedule[qubit + CONTROL_CR] = Channel(center_frequency=cr_frequency)

    def _init_readout_channels(self, qubit: QubitKey):
        readout_frequency = self.params.cavity_frequency[qubit]
        # readout (tx)
        self._schedule[qubit + READOUT_TX] = Channel(
            center_frequency=readout_frequency,
        )
        # readout (rx)
        self._schedule[qubit + READOUT_RX] = Channel(
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
            ctrl_ch.get_timestamp(ctrl_slot) + self._schedule.offset
        )

        # control (ef)
        ctrl_ef_slot = Arbitrary(duration=self.control_window, amplitude=1)
        ctrl_ef_ch = self._ctrl_ef_channel(qubit)
        ctrl_ef_ch.clear()
        ctrl_ef_ch.append(ctrl_ef_slot)
        ctrl_ef_ch.append(Blank(duration=self.readout_window + 4 * T_MARGIN))
        self._ctrl_ef_slots[qubit] = ctrl_ef_slot
        self._ctrl_ef_times[qubit] = (
            ctrl_ef_ch.get_timestamp(ctrl_ef_slot) + self._schedule.offset
        )

        # control (cr)
        ctrl_cr_slot = Arbitrary(duration=self.control_window, amplitude=1)
        ctrl_cr_ch = self._ctrl_cr_channel(qubit)
        ctrl_cr_ch.clear()
        ctrl_cr_ch.append(ctrl_cr_slot)
        ctrl_cr_ch.append(Blank(duration=self.readout_window + 4 * T_MARGIN))
        self._ctrl_cr_slots[qubit] = ctrl_cr_slot
        self._ctrl_cr_times[qubit] = (
            ctrl_cr_ch.get_timestamp(ctrl_cr_slot) + self._schedule.offset
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
            read_tx_ch.get_timestamp(read_tx_slot) + self._schedule.offset
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

        self._triggers = [ports[tx].dac.awg0]

        self._adda_to_channels = {
            ports[tx].dac.awg0: [self._read_tx_channel(qubit) for qubit in self.qubits],
            ports[rx].adc.capt0: [
                self._read_rx_channel(qubit) for qubit in self.qubits
            ],
        }
        for index, control_port in enumerate(self.configs.control_ports):
            port = ports[control_port]
            qubit = self.qubits[index]
            self._adda_to_channels[port.dac.awg0] = [self._ctrl_ef_channel(qubit)]
            self._adda_to_channels[port.dac.awg1] = [self._ctrl_channel(qubit)]
            self._adda_to_channels[port.dac.awg2] = [self._ctrl_cr_channel(qubit)]

    def _reset_tx_waveforms(self):
        for qubit in self.qubits:
            self._ctrl_slots[qubit].iq[:] = 0.0
            self._ctrl_ef_slots[qubit].iq[:] = 0.0
            self._ctrl_cr_slots[qubit].iq[:] = 0.0
            self._read_tx_slots[qubit].iq[:] = 0.0

    def _set_waveforms(
        self,
        control_waveforms: QubitDict[IQArray],
        readout_qubits: list[QubitKey],
    ):
        # reset tx waveforms
        self._reset_tx_waveforms()

        # set control waveforms
        self._set_control_waveforms(control_waveforms)

        # set readout waveforms
        readout_waveforms = self._create_readout_waveforms(readout_qubits)
        self._set_readout_waveforms(readout_qubits, readout_waveforms)

    def _set_control_waveforms(
        self,
        waveforms: QubitDict[IQArray],
    ):
        possible_keys = [
            qubit + suffix
            for qubit in self.qubits
            for suffix in ["", CONTROL_EF, CONTROL_CR]
        ]
        if not all(key in possible_keys for key in waveforms.keys()):
            raise ValueError("Invalid waveform keys.")

        for qubit in self.qubits:
            # control
            if qubit in waveforms:
                values = waveforms[qubit]
                T = len(values) * SAMPLING_PERIOD
                t = self._ctrl_times[qubit]
                self._ctrl_slots[qubit].iq[(-T <= t) & (t < 0)] = values

            # control (ef)
            if qubit + CONTROL_EF in waveforms:
                values = waveforms[qubit + CONTROL_EF]
                T = len(values) * SAMPLING_PERIOD
                t = self._ctrl_ef_times[qubit]
                self._ctrl_ef_slots[qubit].iq[(-T <= t) & (t < 0)] = values

            # control (cr)
            if qubit + CONTROL_CR in waveforms:
                values = waveforms[qubit + CONTROL_CR]
                T = len(values) * SAMPLING_PERIOD
                t = self._ctrl_cr_times[qubit]
                self._ctrl_cr_slots[qubit].iq[(-T <= t) & (t < 0)] = values

    def _set_readout_waveforms(
        self,
        qubits: list[QubitKey],
        waveforms: QubitDict[IQArray],
    ):
        for qubit in qubits:
            values = waveforms[qubit]
            self._read_tx_slots[qubit].iq[:] = values

    def _create_readout_waveforms(self, qubits: list[QubitKey]):
        """Creates readout waveforms for the given qubits."""
        readout_amplitude = self.params.readout_amplitude
        tau = 50
        return {
            qubit: FlatTop(
                width=self.readout_window - tau,
                amplitude=readout_amplitude[qubit],
                tau=tau,
            ).values
            for qubit in qubits
        }
