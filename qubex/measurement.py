"""
A module to represent an measurement.
"""

import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from qubecalib import Qube
from qubecalib.qube import SSB
from qubecalib.pulse import Schedule, Channel, Arbitrary, Blank, Read
from qubecalib.setupqube import run

from .pulse import Waveform, Rcft
from .params import ctrl_freq_dict, ro_freq_dict, ro_ampl_dict

QUBE_ID = "riken_1-07"
QUBITS = ["Q08", "Q09", "Q10", "Q11"]
CTRL = "CTRL_"
READ_TX = "READ_TX_"
READ_RX = "READ_RX_"
T_CONTROL = 10 * 2048  # maximum duration of control [ns]
T_READOUT = 128 * 12  # maximum duration of readout [ns]
T_MARGIN = 128 * 2  # margin between readout and control [ns]
READ_SLICE_RANGE = slice(200, int(200 + 800 / 1.5))
CONTROL_FREQUENCY = ctrl_freq_dict
READOUT_FREQUENCY = ro_freq_dict[QUBE_ID]
READOUT_AMPLITUDE = ro_ampl_dict[QUBE_ID]


MeasuredState = dict[str, complex]


class Measurement:
    """
    A class to represent an measurement.
    """

    def __init__(self, qube_id: str = QUBE_ID):
        self.qubits = QUBITS
        self.qube = self.init_qube(qube_id=qube_id)
        self.schedule = Schedule()
        self.init_channels()
        self.init_schedule()
        self.init_readout_pulses()

    def init_qube(self, qube_id: str):
        """
        Initialize the QuBE.
        """
        qube: Any = Qube.create(f"qube_{qube_id}.yml")

        if qube is None:
            raise ValueError(f"QuBE {qube_id} is not found.")

        # mix.vatt: Attenuator value (max: 0xFFF, min: 0x000)
        # The value saturates around 0xC00, so set it to 0xC00 if you want to get the maximum value.
        # Since the noise may be high at 0xC00, set it to 0x800 for channels except for CR pulses.

        # pylint: disable=no-member

        # Readout TX
        qube.port0.lo.mhz = 12500
        qube.port0.nco.mhz = 2500
        qube.port0.mix.ssb = SSB.LSB
        qube.port0.awg0.nco.mhz = 0
        qube.port0.mix.vatt = 0x800

        # Reatout RX
        qube.port1.nco.mhz = 2500
        qube.port1.adc.capt0.ssb = SSB.LSB
        qube.port1.delay = 128 + 6 * 128  # [ns]

        # Control Q08
        qube.port5.lo.mhz = 10000
        qube.port5.nco.mhz = 2000
        qube.port5.awg0.nco.mhz = 375
        qube.port5.awg1.nco.mhz = 0  # NG
        qube.port5.awg2.nco.mhz = 0
        qube.port5.mix.vatt = 0xC00

        # Control Q09
        qube.port6.lo.mhz = 11000
        qube.port6.nco.mhz = 2250
        qube.port6.awg0.nco.mhz = 375
        qube.port6.awg1.nco.mhz = 0
        qube.port6.awg2.nco.mhz = 375
        qube.port6.mix.vatt = 0x800

        # Control Q10
        qube.port7.lo.mhz = 11000
        qube.port7.nco.mhz = 2500
        qube.port7.awg0.nco.mhz = 375
        qube.port7.awg1.nco.mhz = 0
        qube.port7.awg2.nco.mhz = -375
        qube.port7.mix.vatt = 0x800

        # Control Q11
        qube.port8.lo.mhz = 10000
        qube.port8.nco.mhz = 2250
        qube.port8.awg0.nco.mhz = 375
        qube.port8.awg1.nco.mhz = 0
        qube.port8.awg2.nco.mhz = -375
        qube.port8.mix.vatt = 0xC00

        # pylint: enable=no-member
        return qube

    def init_channels(self):
        """
        Initialize the channels.
        """
        for qubit in self.qubits:
            # control channels
            self.set_control_channel(
                qubit, Channel(center_frequency=CONTROL_FREQUENCY[qubit])
            )
            # readout (tx) channels
            self.set_readout_tx_channel(
                qubit, Channel(center_frequency=READOUT_FREQUENCY[qubit])
            )
            # readout (rx) channels
            self.set_readout_rx_channel(
                qubit, Channel(center_frequency=READOUT_FREQUENCY[qubit])
            )

    def init_schedule(self):
        """
        Initialize the schedule.
        """

        # set the offset of the schedule
        self.schedule.offset = T_CONTROL

        for qubit in self.qubits:
            # control channels
            ctrl_ch: Channel = self.control_channel(qubit)
            ctrl_ch.append(Arbitrary(duration=T_CONTROL))
            ctrl_ch.append(Blank(duration=T_READOUT + 4 * T_MARGIN))

            # readout (tx) channels
            read_tx_ch: Channel = self.readout_tx_channel(qubit)
            read_tx_ch.append(Blank(duration=T_CONTROL))
            read_tx_ch.append(Arbitrary(duration=T_READOUT))
            read_tx_ch.append(Blank(duration=4 * T_MARGIN))

            # readout (rx) channels
            read_rx_ch: Channel = self.readout_rx_channel(qubit)
            read_rx_ch.append(Blank(duration=T_CONTROL - T_MARGIN))
            read_rx_ch.append(Read(duration=T_READOUT + 5 * T_MARGIN))

        durations = [v.duration for k, v in self.schedule.items()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

    def init_readout_pulses(self):
        """
        Initialize the readout pulses.
        """
        for qubit in self.qubits:
            # readout (tx) pulses
            # NOTE: Check the parameters
            pulse = Rcft(
                ampl=READOUT_AMPLITUDE[qubit],
                rise=50,
                flat=int(T_READOUT / 1.5),
                fall=50,
            )
            self.set_readout_waveform(qubit, pulse)

    def measure(self, repeats=10_000, interval=100_000) -> MeasuredState:
        """
        Runs the measurement.
        """
        read_tx_channels = [self.readout_tx_channel(qubit) for qubit in self.qubits]
        read_rx_channels = [self.readout_rx_channel(qubit) for qubit in self.qubits]

        adda_to_channels = {  # NOTE: hi/lo settings
            self.qube.port0.dac.awg0: read_tx_channels,
            self.qube.port1.adc.capt0: read_rx_channels,
            self.qube.port5.dac.awg2: [self.control_channel("Q08")],  # NOTE: AWG1 is NG
            self.qube.port6.dac.awg1: [self.control_channel("Q09")],
            self.qube.port7.dac.awg1: [self.control_channel("Q10")],
            self.qube.port8.dac.awg1: [self.control_channel("Q11")],
        }

        triggers = [self.qube.port0.dac.awg0]

        run(
            self.schedule,
            repeats=repeats,
            interval=interval,
            adda_to_channels=adda_to_channels,
            triggers=triggers,
        )

        state: MeasuredState = self.measured_state()
        return state

    def measured_state(self) -> MeasuredState:
        """
        Returns the measured state (a complex value).
        """
        state = MeasuredState()
        for qubit in self.qubits:
            waveform = self.readout_rx_waveform(qubit)
            state[qubit] = waveform.iq[READ_SLICE_RANGE].mean()
            # save the readout data as a file
            self.save_readout_data(qubit, waveform)
        return state

    def save_readout_data(self, qubit: str, waveform: Waveform):
        """
        Saves the readout data.
        """
        now = datetime.now()
        dir_name = now.strftime("%Y/%m/%d/%H%M%S%f")
        path_str = f"./data/{dir_name}/{READ_RX + qubit}.npy"
        dir_path = os.path.dirname(path_str)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.normpath(path_str)
        data = [waveform.time, waveform.iq]
        np.save(path, data)

    def control_channel(self, qubit: str) -> Channel:
        """
        Returns the control channel of the qubit.
        """
        return self.schedule[CTRL + qubit]

    def readout_tx_channel(self, qubit: str) -> Channel:
        """
        Returns the readout (tx) channel of the qubit.
        """
        return self.schedule[READ_TX + qubit]

    def readout_rx_channel(self, qubit: str) -> Channel:
        """
        Returns the readout (rx) channel of the qubit.
        """
        return self.schedule[READ_RX + qubit]

    def set_control_channel(self, qubit: str, channel: Channel):
        """
        Sets the control channel of the qubit.
        """
        self.schedule[CTRL + qubit] = channel

    def set_readout_tx_channel(self, qubit: str, channel: Channel):
        """
        Sets the readout (tx) channel of the qubit.
        """
        self.schedule[READ_TX + qubit] = channel

    def set_readout_rx_channel(self, qubit: str, channel: Channel):
        """
        Sets the readout (rx) channel of the qubit.
        """
        self.schedule[READ_RX + qubit] = channel

    def control_waveform(self, qubit: str) -> Waveform:
        """
        Returns the control waveform of the channel.
        """
        channel = self.control_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        waveform = Waveform(slot.iq)
        waveform.time = channel.get_timestamp(slot) - self.schedule.offset
        return waveform

    def readout_tx_waveform(self, qubit: str) -> Waveform:
        """
        Returns the readout (tx) waveform of the channel.
        """
        channel: Channel = self.readout_tx_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        waveform = Waveform(slot.iq)
        waveform.time = channel.get_timestamp(slot) - self.schedule.offset
        return waveform

    def readout_rx_waveform(self, qubit: str) -> Waveform:
        """
        Returns the readout (rx) waveform.
        """
        channel: Channel = self.readout_rx_channel(qubit)
        slot: Read = channel.findall(Read)[0]
        if slot.iq is None:
            raise RuntimeError("The readout signal is not recorded.")
        waveform = Waveform(slot.iq)
        waveform.time = channel.get_timestamp(slot) - self.schedule.offset
        return waveform

    def set_control_waveform(self, qubit: str, waveform: Waveform):
        """
        Set the control waveform to the channel.
        """
        channel: Channel = self.control_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        time: np.ndarray = channel.get_timestamp(slot) - self.schedule.offset
        slot.iq[:] = 0j  # initialize
        slot.iq[(-waveform.duration <= time) & (time < 0)] = waveform.iq

    def set_readout_waveform(self, qubit: str, waveform: Waveform):
        """
        Set the readout waveform to the channel.
        """
        channel: Channel = self.readout_tx_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        time: np.ndarray = channel.get_timestamp(slot) - self.schedule.offset
        slot.iq[:] = 0j  # initialize
        slot.iq[(0 <= time) & (time < waveform.duration)] = waveform.iq

    def show_pulse_sequences(self, xlim=(-3.0, 1.5)):
        """
        Shows the pulse sequences.
        """

        # number of qubits
        N = len(self.qubits)

        # initialize the figure
        plt.figure(figsize=(15, 1.5 * (N + 1)))
        plt.subplots_adjust(hspace=0.0)  # no space between subplots

        # the grid is adjusted to the number of qubits (N)
        # the last row (N+1) is for the readout waveform
        gs = gridspec.GridSpec(N + 1, 1)

        # initialize the axes
        # share the x-axis with the first subplot
        axes: list[plt.Axes] = []
        for i, qubit in enumerate(self.qubits):
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
        for i, qubit in enumerate(self.qubits):
            ctrl = self.control_waveform(qubit)
            axes[i].plot(
                ctrl.time * 1e-3,
                ctrl.real,
                label=qubit + " control (real)",
            )
            axes[i].plot(
                ctrl.time * 1e-3,
                ctrl.imag,
                label=qubit + " control (imag)",
            )
            axes[i].legend()
            max_ampl_list.append(np.max(ctrl.ampl))

        # plot the readout pulses
        for i, qubit in enumerate(self.qubits):
            read = self.readout_tx_waveform(qubit)
            axes[N].plot(
                read.time * 1e-3,
                read.ampl,
                label=qubit + " readout (abs)",
                linestyle="dashed",
            )
            axes[N].legend()
            max_ampl_list.append(np.max(read.ampl))

        # set the y-axis range according to the maximum amplitude
        max_ampl = np.max(max_ampl_list)
        for i in range(N + 1):
            axes[i].set_ylim(-1.1 * max_ampl, 1.1 * max_ampl)

        plt.show()
