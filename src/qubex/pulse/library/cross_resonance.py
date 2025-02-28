from __future__ import annotations

from ..pulse_schedule import PulseSchedule
from ..waveform import Waveform
from .flat_top import FlatTop


class CrossResonance(PulseSchedule):
    """
    A class representing the cross-resonance pulse schedule used in quantum experiments.

    Parameters
    ----------
    control_qubit: str
        The label of the control qubit.
    target_qubit: str
        The label of the target qubit.
    cr_amplitude: float
        The amplitude of the cross-resonance pulse.
    cr_duration: float
        The duration of the cross-resonance pulse in nanoseconds.
    cr_ramptime: float
        The ramp duration of the cross-resonance pulse in nanoseconds.
    cr_phase: float
        The phase of the cross-resonance pulse in radians.
    cancel_amplitude: float = 0.0
        The amplitude of the cancel pulse.
    cancel_phase: float = 0.0
        The phase of the cancel pulse in radians.
    echo: bool = False
        If True, the echo pulse is added to the schedule.
    """

    def __init__(
        self,
        control_qubit: str,
        target_qubit: str,
        cr_amplitude: float,
        cr_duration: float,
        cr_ramptime: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        echo: bool = False,
        pi_pulse: Waveform | None = None,
    ):
        cr_ramptime = cr_ramptime or 0.0
        cr_phase = cr_phase or 0.0
        cancel_amplitude = cancel_amplitude or 0.0
        cancel_phase = cancel_phase or 0.0

        cr_label = f"{control_qubit}-{target_qubit}"

        cr_waveform = FlatTop(
            duration=cr_duration,
            amplitude=cr_amplitude,
            tau=cr_ramptime,
            phase=cr_phase,
        )

        cancel_waveform = FlatTop(
            duration=cr_duration,
            amplitude=cancel_amplitude,
            tau=cr_ramptime,
            phase=cancel_phase,
        )

        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.cr_amplitude = cr_amplitude
        self.cr_duration = cr_duration
        self.cr_ramptime = cr_ramptime
        self.cr_phase = cr_phase
        self.cancel_amplitude = cancel_amplitude
        self.cancel_phase = cancel_phase
        self.echo = echo
        self.pi_pulse = pi_pulse
        self.cr_label = cr_label
        self.cr_waveform = cr_waveform
        self.cancel_waveform = cancel_waveform

        with PulseSchedule([cr_label, target_qubit]) as cr:
            cr.add(cr_label, cr_waveform)
            cr.add(target_qubit, cancel_waveform)

        if not echo:
            super().__init__([cr_label, target_qubit])
            self.call(cr)
        else:
            if pi_pulse is None:
                raise ValueError("The pi pulse waveform must be provided.")
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as ecr:
                ecr.call(cr)
                ecr.barrier()
                ecr.add(control_qubit, pi_pulse)
                ecr.barrier()
                ecr.call(cr.scaled(-1))
                ecr.barrier()
                ecr.add(control_qubit, pi_pulse)

            super().__init__([control_qubit, cr_label, target_qubit])
            self.call(ecr)
