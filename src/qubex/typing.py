from __future__ import annotations

from typing import Callable, Mapping, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from .pulse import PulseSchedule, Waveform

T_co = TypeVar("T_co", covariant=True)

TargetMap = Mapping[str, T_co]

IQArray = Union[
    list,
    list[complex],
    list[float],
    NDArray[np.complex128],
    NDArray[np.float64],
]

ParametricWaveformDict = Callable[..., TargetMap[Waveform]]
ParametricPulseSchedule = Callable[..., PulseSchedule]
