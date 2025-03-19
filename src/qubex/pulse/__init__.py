from .blank import Blank
from .library import (
    CPMG,
    CrossResonance,
    Drag,
    FlatTop,
    Gaussian,
    RaisedCosine,
    Rect,
)
from .phase_shift import PhaseShift, VirtualZ
from .pulse import Pulse
from .pulse_array import PulseArray, PulseSequence
from .pulse_schedule import PulseChannel, PulseSchedule
from .waveform import Waveform

__all__ = [
    "Blank",
    "CPMG",
    "CrossResonance",
    "Drag",
    "FlatTop",
    "Gaussian",
    "PhaseShift",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "PulseSequence",
    "RaisedCosine",
    "Rect",
    "VirtualZ",
    "Waveform",
]


def set_sampling_period(dt: float):
    Waveform.SAMPLING_PERIOD = dt


def get_sampling_period():
    return Waveform.SAMPLING_PERIOD
