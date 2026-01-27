from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from qubex.pulse import PulseSchedule, Waveform

T_co = TypeVar("T_co", covariant=True)

TargetMap: TypeAlias = Mapping[str, T_co]

IQArray: TypeAlias = (
    list[complex] | list[float] | NDArray[np.complexfloating] | NDArray[np.floating]
)

ParametricWaveformDict: TypeAlias = Callable[..., TargetMap[Waveform]]
ParametricPulseSchedule: TypeAlias = Callable[..., PulseSchedule]
