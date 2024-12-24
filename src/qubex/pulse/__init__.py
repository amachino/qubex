from .library import (
    CPMG,
    Drag,
    DragCos,
    DragGauss,
    FlatTop,
    Gauss,
    Gaussian,
    RaisedCosine,
    Rect,
)
from .pulse import Blank, Pulse
from .pulse_schedule import PulseSchedule
from .pulse_sequence import PhaseShift, PulseSequence, VirtualZ
from .waveform import Waveform

__all__ = [
    "Blank",
    "CPMG",
    "Drag",
    "DragCos",
    "DragGauss",
    "FlatTop",
    "Gauss",
    "Gaussian",
    "PhaseShift",
    "Pulse",
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
