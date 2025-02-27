from .library import (
    CPMG,
    CrossResonance,
    Drag,
    FlatTop,
    Gaussian,
    RaisedCosine,
    Rect,
)
from .pulse import Blank, Pulse
from .pulse_array import PhaseShift, PulseArray, PulseSequence, VirtualZ
from .pulse_schedule import PulseSchedule
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
