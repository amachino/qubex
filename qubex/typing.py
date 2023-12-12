from typing import Callable, Mapping, TypeVar

import numpy as np
from numpy.typing import NDArray

from .pulse import Waveform

QubitKey = str
QubitValue = TypeVar("QubitValue")
QubitDict = Mapping[QubitKey, QubitValue]

IQValue = complex
IQArray = NDArray[np.complex128]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

PortConfigs = dict[str, dict[str, int]]

ParametricWaveform = Callable[..., Waveform]
