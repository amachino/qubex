"""
A module to represent an measurement.
"""

from datetime import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import qubecalib
from qubecalib.ui import QubeControl
from qubecalib.pulse import Schedule, Channel, Arbitrary, Blank, Read
from qubecalib.setupqube import run

from .pulse import Pulse, Rcft
from .params import ctrl_freq_dict, ro_freq_dict, ro_ampl_dict

QUBE_ID = "riken107"
QUBE_CONFIG_FILE = "qube_riken_1-07.yml"
QUBITS = ["Q08", "Q09", "Q10", "Q11"]
CTRL = ""
RO_SEND = "RO_SEND_"
RO_RETURN = "RO_RETURN_"
T_CONTROL = 10 * 2048  # maximum duration of control [ns]
T_READOUT = 128 * 12  # maximum duration of readout [ns]
T_MARGIN = 128 * 2  # margin between readout and control [ns]
READ_SLICE_RANGE = slice(200, int(200 + 800 / 1.5))
CTRL_FREQ = ctrl_freq_dict
RO_FREQ = ro_freq_dict[QUBE_ID]
RO_AMPL = ro_ampl_dict[QUBE_ID]


MeasuredSignal = dict[str, complex]


class Measurement:
    """
    A class to represent an measurement.
    """

    def __init__(self):
        self.qubits = QUBITS
        self.qube = self.init_qube()
        self.schedule = Schedule()
        self.init_channels()
        self.init_schedule()
        self.set_readout_pulse()

    def init_qube(self):
        """
        Initialize the QuBE.
        """
        qubecalib.ui.MATPLOTLIB_PYPLOT = plt
        qube = QubeControl(QUBE_CONFIG_FILE).qube

        # mix.vatt: Attenuator value (max: 0xFFF, min: 0x000)
        # The value saturates around 0xC00, so set it to 0xC00 if you want to get the maximum value.
        # Since the noise may be high at 0xC00, set it to 0x800 for channels except for CR pulses.

        # TODO: Clarify the meaning of the following parameters

        # RO send
        qube.port0.lo.mhz = 12500
        qube.port0.nco.mhz = 2500
        qube.port0.mix.ssb = qubecalib.qube.SSB.LSB
        qube.port0.awg0.nco.mhz = 0
        qube.port0.mix.vatt = 0x800

        # RO return
        qube.port1.nco.mhz = 2500
        qube.port1.adc.capt0.ssb = qubecalib.qube.SSB.LSB
        qube.port1.delay = 128 + 6 * 128  # [ns]

        # Q08
        qube.port5.lo.mhz = 10000
        qube.port5.nco.mhz = 2000
        qube.port5.awg0.nco.mhz = 375
        qube.port5.awg1.nco.mhz = 0  # NG
        qube.port5.awg2.nco.mhz = 0
        qube.port5.mix.vatt = 0xC00

        # Q09
        qube.port6.lo.mhz = 11000
        qube.port6.nco.mhz = 2250
        qube.port6.awg0.nco.mhz = 375
        qube.port6.awg1.nco.mhz = 0
        qube.port6.awg2.nco.mhz = 375
        qube.port6.mix.vatt = 0x800

        # Q10
        qube.port7.lo.mhz = 11000
        qube.port7.nco.mhz = 2500
        qube.port7.awg0.nco.mhz = 375
        qube.port7.awg1.nco.mhz = 0
        qube.port7.awg2.nco.mhz = -375
        qube.port7.mix.vatt = 0x800

        # Q11
        qube.port8.lo.mhz = 10000
        qube.port8.nco.mhz = 2250
        qube.port8.awg0.nco.mhz = 375
        qube.port8.awg1.nco.mhz = 0
        qube.port8.awg2.nco.mhz = -375
        qube.port8.mix.vatt = 0xC00

        return qube

    def init_channels(self):
        """
        Initialize the channels.
        """
        for qubit in self.qubits:
            # control channels
            self.set_control_channel(qubit, Channel(center_frequency=CTRL_FREQ[qubit]))
            # readout (send) channels
            self.set_ro_send_channel(qubit, Channel(center_frequency=RO_FREQ[qubit]))
            # readout (return) channels
            self.set_ro_return_channel(qubit, Channel(center_frequency=RO_FREQ[qubit]))

    def init_schedule(self):
        """
        Initialize the schedule.
        """

        # set the offset of the schedule
        self.schedule.offset = T_CONTROL

        # TODO: Clarify the meaning of the following time parameters
        for qubit in self.qubits:
            # control channels
            control_ch: Channel = self.control_channel(qubit)
            control_ch.append(Arbitrary(duration=T_CONTROL))
            control_ch.append(Blank(duration=T_READOUT + 4 * T_MARGIN))

            # readout (send) channels
            ro_send_ch: Channel = self.ro_send_channel(qubit)
            ro_send_ch.append(Blank(duration=T_CONTROL))
            ro_send_ch.append(Arbitrary(duration=T_READOUT))
            ro_send_ch.append(Blank(duration=4 * T_MARGIN))

            # readout (return) channels
            ro_return_ch: Channel = self.ro_return_channel(qubit)
            ro_return_ch.append(Blank(duration=T_CONTROL - T_MARGIN))
            ro_return_ch.append(Read(duration=T_READOUT + 5 * T_MARGIN))

        durations = [v.duration for k, v in self.schedule.items()]
        assert len(set(durations)) == 1, "All channels must have the same duration."

    def set_readout_pulse(self):
        """
        Initialize the readout pulses.
        """
        for qubit in self.qubits:
            # readout (send) pulses
            # TODO: Clarify the meaning of the following parameters
            pulse = Rcft(
                ampl=RO_AMPL[qubit],
                rise=50,
                flat=int(T_READOUT / 1.5),
                fall=50,
            )
            self.set_ro_send_pulse(qubit, pulse)

    # TODO: Clarify the meaning of the following parameters
    def measure(self, repeats=10_000, interval=100_000) -> MeasuredSignal:
        """
        Runs the measurement.
        """
        ro_send_channels = [self.ro_send_channel(qubit) for qubit in self.qubits]
        ro_return_channels = [self.ro_return_channel(qubit) for qubit in self.qubits]

        adda_to_channels = {  # TODO: hi/lo settings
            self.qube.port0.dac.awg0: ro_send_channels,
            self.qube.port1.adc.capt0: ro_return_channels,
            self.qube.port5.dac.awg2: [self.control_channel("Q08")],
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

        signal: MeasuredSignal = self.measured_signal()
        return signal

    def measured_signal(self) -> MeasuredSignal:
        """
        Returns the measured signal.
        """
        signal = MeasuredSignal()
        for qubit in self.qubits:
            pulse = self.ro_return_pulse(qubit)
            signal[qubit] = pulse.waveform[READ_SLICE_RANGE].mean()
            # save the readout data as a file
            self.save_readout_pulse(qubit, pulse)
        return signal

    def save_readout_pulse(self, qubit: str, pulse: Pulse):
        """
        Saves the readout (return) pulse.
        """
        now = datetime.now()
        dir_name = now.strftime("%Y/%m/%d/%H%M%S%f")
        path_str = f"./data/{dir_name}/{RO_RETURN + qubit}.npy"
        dir_path = os.path.dirname(path_str)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.normpath(path_str)
        data = [pulse.time, pulse.waveform]
        np.save(path, data)

    def control_channel(self, qubit: str) -> Channel:
        """
        Returns the control channel of the qubit.
        """
        return self.schedule[CTRL + qubit]

    def ro_send_channel(self, qubit: str) -> Channel:
        """
        Returns the readout (send) channel of the qubit.
        """
        return self.schedule[RO_SEND + qubit]

    def ro_return_channel(self, qubit: str) -> Channel:
        """
        Returns the readout (return) channel of the qubit.
        """
        return self.schedule[RO_RETURN + qubit]

    def set_control_channel(self, qubit: str, channel: Channel):
        """
        Sets the control channel of the qubit.
        """
        self.schedule[CTRL + qubit] = channel

    def set_ro_send_channel(self, qubit: str, channel: Channel):
        """
        Sets the readout (send) channel of the qubit.
        """
        self.schedule[RO_SEND + qubit] = channel

    def set_ro_return_channel(self, qubit: str, channel: Channel):
        """
        Sets the readout (return) channel of the qubit.
        """
        self.schedule[RO_RETURN + qubit] = channel

    def control_pulse(self, qubit: str) -> Pulse:
        """
        Returns the control pulse of the channel.
        """
        channel = self.control_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        pulse = Pulse(slot.iq)
        pulse.time = channel.get_timestamp(slot) - self.schedule.offset
        return pulse

    def ro_send_pulse(self, qubit: str) -> Pulse:
        """
        Returns the readout (send) pulse of the channel.
        """
        channel: Channel = self.ro_send_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        pulse = Pulse(slot.iq)
        pulse.time = channel.get_timestamp(slot) - self.schedule.offset
        return pulse

    def ro_return_pulse(self, qubit: str) -> Pulse:
        """
        Returns the readout (return) pulse of the channel.
        """
        channel: Channel = self.ro_return_channel(qubit)
        slot: Arbitrary = channel.findall(Read)[0]
        if slot.iq is None:
            raise RuntimeError("The readout signal is not recorded.")
        pulse = Pulse(slot.iq)
        pulse.time = channel.get_timestamp(slot) - self.schedule.offset
        return pulse

    def set_control_pulse(self, qubit: str, pulse: Pulse):
        """
        Set the control pulse to the channel.
        """
        channel: Channel = self.control_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        time: np.ndarray = channel.get_timestamp(slot) - self.schedule.offset
        slot.iq[:] = 0j  # initialize
        slot.iq[(-pulse.duration <= time) & (time < 0)] = pulse.waveform

    def set_ro_send_pulse(self, qubit: str, pulse: Pulse):
        """
        Set the transmit pulse to the channel.
        """
        channel: Channel = self.ro_send_channel(qubit)
        slot: Arbitrary = channel.findall(Arbitrary)[0]
        time: np.ndarray = channel.get_timestamp(slot) - self.schedule.offset
        slot.iq[:] = 0j  # initialize
        slot.iq[(0 <= time) & (time < pulse.duration)] = pulse.waveform

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
            ctrl_pulse = self.control_pulse(qubit)
            axes[i].plot(
                ctrl_pulse.time * 1e-3,
                ctrl_pulse.real,
                label=qubit + " ctrl (real)",
            )
            axes[i].plot(
                ctrl_pulse.time * 1e-3,
                ctrl_pulse.imag,
                label=qubit + " ctrl (imag)",
            )
            axes[i].legend()
            max_ampl_list.append(np.max(ctrl_pulse.ampl))

        # plot the readout pulses
        for i, qubit in enumerate(self.qubits):
            ro_pulse = self.ro_send_pulse(qubit)
            axes[N].plot(
                ro_pulse.time * 1e-3,
                ro_pulse.ampl,
                label=qubit + " readout (abs)",
                linestyle="dashed",
            )
            axes[N].legend()
            max_ampl_list.append(np.max(ro_pulse.ampl))

        # set the y-axis range according to the maximum amplitude
        max_ampl = np.max(max_ampl_list)
        for i in range(N + 1):
            axes[i].set_ylim(-1.1 * max_ampl, 1.1 * max_ampl)

        plt.show()
