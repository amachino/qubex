from __future__ import annotations

from typing import Literal

from typing_extensions import TypeAlias

RampType: TypeAlias = Literal[
    "Gaussian",
    "RaisedCosine",
    "Sintegral",
    "Bump",
    "Squad",
]
