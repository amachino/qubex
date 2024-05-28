from __future__ import annotations

from typing import Mapping, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

T_co = TypeVar("T_co", covariant=True)
TargetMap = Mapping[str, T_co]

IQArray = Union[
    list[complex],
    list[float],
    NDArray[np.complex128],
    NDArray[np.float64],
]
