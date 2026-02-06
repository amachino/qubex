"""Ramp type helpers for pulse shaping."""

from __future__ import annotations

from typing import Literal, TypeAlias

RampType: TypeAlias = Literal[
    "Gaussian",
    "RaisedCosine",
    "Sintegral",
    "Bump",
    "Squad",
]
