from __future__ import annotations

from typing import Literal

from ..blank import Blank
from ..pulse_array import PulseArray
from ..pulse_schedule import PulseSchedule
from ..waveform import Waveform
from .flat_top import FlatTop, MultiDerivativeFlatTop


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
    cr_beta: float
        The DRAG correction coefficient for the cross-resonance pulse.
    cancel_amplitude: float = 0.0
        The amplitude of the cancel pulse.
    cancel_phase: float = 0.0
        The phase of the cancel pulse in radians.
    cancel_beta: float = 0.0
        The DRAG correction coefficient for the cancel pulse.
    echo: bool = False
        If True, the echo pulse is added to the schedule.
    ramp_type: Literal["Gaussian", "RaisedCosine", "Sintegral", "Bump"] = "RaisedCosine",
        The type of the ramp function used in the pulse.
    """

    def __init__(
        self,
        control_qubit: str,
        target_qubit: str,
        cr_amplitude: float,
        cr_duration: float,
        cr_ramptime: float | None = None,
        cr_phase: float | None = None,
        cr_beta: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_beta: float | None = None,
        echo: bool = False,
        pi_pulse: Waveform | None = None,
        pi_margin: float | None = None,
        ramp_type: Literal[
            "Gaussian",
            "RaisedCosine",
            "Sintegral",
            "Bump",
        ] = "RaisedCosine",
    ):
        cr_ramptime = cr_ramptime or 0.0
        cr_phase = cr_phase or 0.0
        cr_beta = cr_beta or 0.0
        cancel_amplitude = cancel_amplitude or 0.0
        cancel_phase = cancel_phase or 0.0
        cancel_beta = cancel_beta or 0.0
        pi_margin = pi_margin or 0.0

        cr_label = f"{control_qubit}-{target_qubit}"

        cr_waveform = FlatTop(
            duration=cr_duration,
            amplitude=cr_amplitude,
            tau=cr_ramptime,
            phase=cr_phase,
            beta=cr_beta,
            type=ramp_type,
        )

        cancel_waveform = FlatTop(
            duration=cr_duration,
            amplitude=cancel_amplitude,
            tau=cr_ramptime,
            phase=cancel_phase,
            beta=cancel_beta,
            type=ramp_type,
        )

        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.cr_amplitude = cr_amplitude
        self.cr_duration = cr_duration
        self.cr_ramptime = cr_ramptime
        self.cr_phase = cr_phase
        self.cr_beta = cr_beta
        self.cancel_amplitude = cancel_amplitude
        self.cancel_phase = cancel_phase
        self.cancel_beta = cancel_beta
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
            if pi_margin > 0:
                margin = Blank(duration=pi_margin)
                pi_pulse = PulseArray([margin, pi_pulse, margin])
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


class MultiDerivativeCrossResonance(PulseSchedule):
    """
    A class representing the multi-derivative cross-resonance pulse schedule.

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
    cr_betas: dict[int, float]
        The multi-derivative pulse correction coefficients for the cross-resonance pulse.
    cr_power: int = 2
        The power of the sine integral for the cross-resonance pulse.
    cancel_amplitude: float = 0.0
        The amplitude of the cancel pulse.
    cancel_phase: float = 0.0
        The phase of the cancel pulse in radians.
    cancel_betas: dict[int, float] = None
        The multi-derivative pulse correction coefficients for the cancel pulse.
    cancel_power: int = 2
        The power of the sine integral for the cancel pulse.
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
        cr_betas: dict[int, float] | None = None,
        cr_power: int = 2,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_betas: dict[int, float] | None = None,
        cancel_power: int = 2,
        echo: bool = False,
        pi_pulse: Waveform | None = None,
        pi_margin: float | None = None,
    ):
        cr_ramptime = cr_ramptime or 0.0
        cr_phase = cr_phase or 0.0
        cr_betas = cr_betas or {}
        cancel_amplitude = cancel_amplitude or 0.0
        cancel_phase = cancel_phase or 0.0
        cancel_betas = cancel_betas or {}
        pi_margin = pi_margin or 0.0

        cr_label = f"{control_qubit}-{target_qubit}"

        cr_waveform = MultiDerivativeFlatTop(
            duration=cr_duration,
            amplitude=cr_amplitude,
            tau=cr_ramptime,
            phase=cr_phase,
            betas=cr_betas,
            power=cr_power,
        )

        cancel_waveform = MultiDerivativeFlatTop(
            duration=cr_duration,
            amplitude=cancel_amplitude,
            tau=cr_ramptime,
            phase=cancel_phase,
            betas=cancel_betas,
            power=cancel_power,
        )

        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.cr_amplitude = cr_amplitude
        self.cr_duration = cr_duration
        self.cr_ramptime = cr_ramptime
        self.cr_phase = cr_phase
        self.cr_betas = cr_betas
        self.cr_power = cr_power
        self.cancel_amplitude = cancel_amplitude
        self.cancel_phase = cancel_phase
        self.cancel_betas = cancel_betas
        self.cancel_power = cancel_power
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
            if pi_margin > 0:
                margin = Blank(duration=pi_margin)
                pi_pulse = PulseArray([margin, pi_pulse, margin])
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
