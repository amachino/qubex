from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .pulse import PulseSchedule, Waveform

T_co = TypeVar("T_co", covariant=True)

TargetMap = Mapping[str, T_co]

IQArray = list[complex] | list[float] | NDArray[np.complex128] | NDArray[np.float64]

ParametricWaveformDict = Callable[..., TargetMap[Waveform]]
ParametricPulseSchedule = Callable[..., PulseSchedule]
