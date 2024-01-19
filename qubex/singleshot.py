from collections import namedtuple
import datetime
import numpy as np
import qubecalib
from e7awgsw import AwgCtrl, CaptureCtrl, CaptureParam, CaptureModule, AWG
from qubecalib.setupqube import _conv_to_e7awgsw


class PulseConverter(object):
    Container = namedtuple(
        "Container",
        [
            "awg_to_wavesequence",
            "capt_to_captparam",
            "capt_to_mergedchannel",
            "adda_to_channels",
        ],
    )

    @classmethod
    def conv(cls, channels, offset=0, interval=0):
        r = _conv_to_e7awgsw(
            adda_to_channels=channels,
            offset=offset,
            repeats=1,
            interval=interval,
            trigger_awg=None,
        )
        func = lambda v: {
            k2: (cls.duplicate_captparam(v2),)
            for k2, v2 in v["capt_to_captparam"].items()
        }
        return dict(
            [
                (
                    k,
                    cls.Container(
                        v["awg_to_wavesequence"],
                        func(v),
                        v["capt_to_mergedchannel"],
                        channels,
                    ),
                )
                for k, v in r.items()
            ]
        )

    @classmethod
    def duplicate_captparam(cls, cp, repeats=1, interval=0, delay=0):
        p = CaptureParam()
        p.num_integ_sections = cp.num_integ_sections
        p.capture_delay = cp.capture_delay
        for i in range(repeats):
            for s in cp.sum_section_list:
                p.add_sum_section(*s)
        return p


class Recv(CaptureCtrl):
    def __init__(self, *module_params_pair):
        argparse = lambda module, *params: (module, params)
        cond = lambda o: isinstance(o, tuple) or isinstance(o, list)
        arg = tuple(
            [
                argparse(o[0], *(o[1] if cond(o[1]) else (o[1],)))
                for o in module_params_pair
            ]
        )

        # typing で書くのがいまどき？
        if not [isinstance(m, qubecalib.qube.CPT) for m, l in arg] == len(arg) * [True]:
            raise TypeError(
                "1st element of each tuple should be qubecalib.meas.CPT instance."
            )

        if not [arg[0][0].port.qube == m.port.qube for m, l in arg] == len(arg) * [
            True
        ]:
            raise Exception(
                "The qube that owns the CaptureModule candidates in the arguments must all be identical."
            )

        if not [len(l) < 5 for m, l in arg] == len(arg) * [True]:
            raise Exception(
                "Each CaptureParameter list in the argument must have no longer than 4 elements."
            )

        super().__init__(arg[0][0].port.qube.ipfpga)

        self._trigger = None  # obsoleted
        self.modules = [m for m, l in arg]
        self.units = sum([self.assign_param_to_unit(m, l) for m, l in arg], [])

    def assign_param_to_unit(self, module, params):
        m = module
        units = [
            u
            for u in CaptureModule.get_units(
                m if isinstance(m, CaptureModule) else m.id
            )[: len(params)]
        ]
        self.initialize(*units)
        for u, p in zip(units, params):
            self.set_capture_params(u, p)

        return units

    def start(self, timeout=30):
        u = self.units
        self.start_capture_units(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)

    def wait_for_trigger(self, awg):
        trig = awg if isinstance(awg, AWG) else awg.id
        for m in self.modules:
            self.select_trigger_awg(m.id, trig)
        self.enable_start_trigger(*self.units)

    def wait_for_capture(self, timeout=30):
        u = self.units
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)

    def get_data(self, unit):
        n = self.num_captured_samples(unit)
        c = np.array(self.get_capture_data(unit, n))
        return c[:, 0] + 1j * c[:, 1]

    def check_err(self, *units):
        e = super().check_err(*units)
        if any(e):
            raise IOError("CaptureCtrl error.")

    def wait(self, timeout=30):  # obsoleted
        u = self.units
        self.enable_start_trigger(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)

    @property
    def trigger(self):  # obsoleted
        return self._trigger

    @trigger.setter
    def trigger(self, awg):  # obsoleted
        self._trigger = awg if isinstance(awg, AWG) else awg.id
        for m in self.modules:
            self.select_trigger_awg(m.id, self._trigger)


class CaptMemory(CaptureCtrl):  # for data access
    def get_data(self, unit):
        n = self.num_captured_samples(unit)
        c = np.array(self.get_capture_data(unit, n))
        return c[:, 0] + 1j * c[:, 1]


class Send(AwgCtrl):
    def __init__(self, *awg_seqfactory_pair):
        arg = awg_seqfactory_pair

        # typing で書くのがいまどき？
        if not [
            isinstance(a, qubecalib.qube.AWG)
            and isinstance(s, qubecalib.meas.WaveSequenceFactory)
            for a, s in arg
        ] == len(arg) * [True]:
            raise TypeError(
                "Element type of each tuple should be (qubecalib.qube.AWG, qubecalib.meas.WaveSequenceFactory)."
            )

        if not [arg[0][0].port.qube == a.port.qube for a, s in arg] == len(arg) * [
            True
        ]:
            raise Exception(
                "The qube that owns the AWG candidates in the arguments must all be identical."
            )

        super().__init__(arg[0][0].port.qube.ipfpga)

        self.awgs = awgs = [a for a, s in arg]

        self.initialize(*[a.id for a in awgs])
        for a, s in arg:
            self.set_wave_sequence(a.id, s.sequence)

    def start(self):
        a = [a.id for a in self.awgs]
        self.terminate_awgs(*a)
        self.clear_awg_stop_flags(*a)
        self.start_awgs(*a)

    def wait_for_sequencer(self, timeout=30):
        a = [a.id for a in self.awgs]
        self.terminate_awgs(*a)
        self.clear_awg_stop_flags(*a)
        print("wait:", datetime.datetime.now())
        print(
            "wait for started by sequencer for {}".format(self.awgs[0].port.qube.ipfpga)
        )
        self.wait_for_awgs_to_stop(timeout, *a)
        print("awg done:", datetime.datetime.now())
        print("end")


def _singleshot(adda_to_channels, triggers, repeats, timeout=30, interval=50000):
    c = PulseConverter.conv(adda_to_channels, interval=50000)

    units = {}
    for qube in tuple(c.keys()):
        trigger = [o for o in triggers if o.port.qube == qube][0]
        units[qube] = singleshot_singleqube(c[qube], trigger, repeats, timeout)

    return units


def singleshot_singleqube(pulse, trigger, repeats, timeout=30, interval=50000):
    arg_send = tuple([(a, w) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = tuple([(c, p) for c, p in pulse.capt_to_captparam.items()])

    for a, w in arg_send:
        w.num_repeats = repeats
    arg_recv = tuple(
        [
            (
                c,
                tuple(
                    [
                        PulseConverter.duplicate_captparam(pi, repeats=repeats)
                        for pi in p
                    ]
                ),
            )
            for c, p in arg_recv
        ]
    )

    with Send(*arg_send) as s, Recv(*arg_recv) as r:
        r.wait_for_trigger(trigger)
        s.start()
        r.wait_for_capture(timeout=timeout)

        units = r.units

    captm = arg_recv[0][0]
    for channel in pulse.adda_to_channels[captm]:
        singleshot_get_data(captm, channel, repeats)

    return units


def singleshot_get_data(captm, channel, repeats):
    unit = qubecalib.meas.CaptureModule.get_units(captm.id)[0]  # <- とりあえず OK
    slot = channel.findall(qubecalib.pulse.Read)[0]
    with CaptMemory(captm.port.qube.ipfpga) as m:
        v = m.get_data(unit)
        v = v.reshape(repeats, int(len(v) / repeats))
        t = np.arange(0, len(v[0])) / qubecalib.meas.CaptureCtrl.SAMPLING_RATE
        v *= np.exp(
            -1j
            * 2
            * np.pi
            * captm.modulation_frequency(channel.center_frequency * 1e-6)
            * 1e6
            * t
        )
        d = slot.duration
        m = max([s.sampling_rate for s in channel])
        slot.iq = v
