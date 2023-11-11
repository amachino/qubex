from typing import TypeVar, Literal

import numpy as np
from numpy.typing import NDArray

QubitLabel = str

T = TypeVar("T")
QubitDict = dict[QubitLabel, T]

IQValue = complex

IQArray = NDArray[np.complex128]

ReadoutTxPort = Literal["port0", "port13"]
ReadoutRxPort = Literal["port1", "port12"]
ReadoutPorts = tuple[ReadoutTxPort, ReadoutRxPort]
