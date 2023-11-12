from typing import TypeVar, Literal, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .pulse import Waveform

QubitKey = str
QubitValue = TypeVar("QubitValue")
QubitDict = Mapping[QubitKey, QubitValue]

IQValue = complex
IQArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

ReadoutTxPort = Literal["port0", "port13"]
ReadoutRxPort = Literal["port1", "port12"]
ReadoutPorts = tuple[ReadoutTxPort, ReadoutRxPort]

ParametricWaveform = Callable[..., Waveform]
