"""
singleshot.py

This module provides a function to execute singleshot measurement.
Assume that the qube is in standalone mode.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# mypy: disable-error-code="import-untyped"
from e7awgsw import AwgCtrl, CaptureCtrl, CaptureModule, CaptureParam, CaptureUnit
from qubecalib.meas import WaveSequenceFactory
from qubecalib.pulse import Channel, Read
from qubecalib.qube import AWG, CPT, QubeTypeA
from qubecalib.setupqube import _conv_to_e7awgsw


class Send(AwgCtrl):
    def __init__(
        self,
        args: list[tuple[AWG, WaveSequenceFactory]],
    ):
        qube: QubeTypeA = args[0][0].port.qube
        super().__init__(qube.ipfpga)

        self.awg_id_list = [awg.id for awg, _ in args]
        self.initialize(*self.awg_id_list)
        for awg, seq in args:
            self.set_wave_sequence(awg.id, seq.sequence)

    def start(self):
        ids = self.awg_id_list
        self.terminate_awgs(*ids)
        self.clear_awg_stop_flags(*ids)
        self.start_awgs(*ids)


class Recv(CaptureCtrl):
    def __init__(
        self,
        args: list[tuple[CPT, CaptureParam]],
    ):
        qube: QubeTypeA = args[0][0].port.qube
        super().__init__(qube.ipfpga)

        self.captm_id_list = [captm.id for captm, _ in args]
        self.capt_units = CaptureModule.get_units(*self.captm_id_list)
        self.initialize(*self.capt_units)
        for captm, param in args:
            for unit in CaptureModule.get_units(captm.id):
                self.set_capture_params(unit, param)

    def wait_for_trigger(self, awg: AWG):
        for captm in self.captm_id_list:
            self.select_trigger_awg(captm, awg.id)
        self.enable_start_trigger(*self.capt_units)

    def wait_for_capture(self, timeout=30):
        units = self.capt_units
        self.wait_for_capture_units_to_stop(timeout, *units)
        self.check_err(*units)

    def check_err(self, *units):
        err = super().check_err(*units)
        if err:
            raise IOError("CaptureCtrl error.")


def singleshot(
    adda_to_channels: dict[AWG | CPT, list[Channel]],
    triggers: list[AWG],
    readout_range: slice = slice(None),
    shots: int = 1024,
    interval: int = 150_000,
    timeout: int = 30,
):
    # create config object for e7awgsw
    qube_to_e7awgsw: dict[QubeTypeA, dict[str, dict]] = _conv_to_e7awgsw(
        adda_to_channels=adda_to_channels,
        offset=0,
        repeats=shots,
        interval=interval,
        trigger_awg=None,
    )
    qubes = list(qube_to_e7awgsw.keys())
    if len(qubes) != 1:
        raise ValueError("The number of qubes must be 1 for standalone mode.")
    if len(triggers) != 1:
        raise ValueError("The number of triggers must be 1 for standalone mode.")

    qube = qubes[0]
    config = qube_to_e7awgsw[qube]
    trigger: AWG = triggers[0]

    awg_to_wavesequence: dict[AWG, WaveSequenceFactory] = config["awg_to_wavesequence"]
    capt_to_captparam: dict[CPT, CaptureParam] = config["capt_to_captparam"]

    # modify CaptureParam for singleshot
    for captparam in capt_to_captparam.values():
        captparam.sel_dsp_units_to_enable()

    # args for Send and Recv
    arg_send = list(awg_to_wavesequence.items())
    arg_recv = list(capt_to_captparam.items())

    # execute singleshot
    with Send(arg_send) as send, Recv(arg_recv) as recv:
        recv.wait_for_trigger(trigger)
        send.start()
        recv.wait_for_capture(timeout)

    # capture module
    capt_module = arg_recv[0][0]

    # capture units
    capt_units: list[CaptureUnit] = CaptureModule.get_units(capt_module.id)

    # readout channels
    read_channels = adda_to_channels[capt_module]

    # store readout data
    for capt_unit, read_channel in zip(capt_units, read_channels):
        # demodulate readout data
        read_slot = read_channel.findall(Read)[0]
        frequency = read_channel.center_frequency * 1e-6
        data = _demodulate_readout_data(
            capt_module=capt_module,
            capt_unit=capt_unit,
            frequency=frequency,
            readout_range=readout_range,
            shots=shots,
        )
        # store demodulated data to readout slot
        read_slot.iq = data


def _demodulate_readout_data(
    capt_module: CPT,
    capt_unit: CaptureUnit,
    frequency: float,
    readout_range: slice,
    shots: int,
) -> npt.NDArray[np.complex128]:
    with CaptureCtrl(capt_module.port.qube.ipfpga) as capt_ctrl:
        # get the number of captured samples
        n_samples = capt_ctrl.num_captured_samples(capt_unit)
        # get the captured data from the capture unit
        capture_data = np.array(capt_ctrl.get_capture_data(capt_unit, n_samples))
        # convert to complex array
        values = capture_data[:, 0] + 1j * capture_data[:, 1]
        # reshape the array to separate shots
        values = values.reshape(shots, len(values) // shots)
        # calculate time axis
        times = np.arange(0, len(values[0])) / CaptureCtrl.SAMPLING_RATE
        # calculate modulated frequency (difference from NCO frequency)
        modulated_frequency = capt_module.modulation_frequency(frequency)
        # demodulate the readout waveform
        values *= np.exp(-1j * 2 * np.pi * modulated_frequency * 1e6 * times)
        # average the waveform to get a I/Q value
        values = values[:, readout_range].mean(axis=1)
        # return the I/Q value
        return values