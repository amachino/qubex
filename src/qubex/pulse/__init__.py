from .blank import Blank
from .library import (
    CPMG,
    XY4,
    Bump,
    CrossResonance,
    Drag,
    FlatTop,
    Gaussian,
    MultiDerivativeCrossResonance,
    RaisedCosine,
    RampType,
    Rect,
    Sintegral,
)
from .phase_shift import PhaseShift, VirtualZ
from .pulse import Pulse
from .pulse_array import PulseArray
from .pulse_schedule import PulseChannel, PulseSchedule
from .waveform import Waveform

__all__ = [
    "Blank",
    "Bump",
    "CPMG",
    "CrossResonance",
    "Drag",
    "FlatTop",
    "Gaussian",
    "MultiDerivativeCrossResonance",
    "PhaseShift",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "RaisedCosine",
    "RampType",
    "Rect",
    "Sintegral",
    "VirtualZ",
    "Waveform",
    "XY4",
]


def set_sampling_period(dt: float):
    Waveform.SAMPLING_PERIOD = dt


def get_sampling_period():
    return Waveform.SAMPLING_PERIOD
