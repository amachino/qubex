"""Shared typing aliases for Qubex."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from qxcore import Frequency, Time, Value
from qxpulse.typing import IQArray as _IQArray

if TYPE_CHECKING:
    from qxpulse import PulseSchedule, Waveform

T_co = TypeVar("T_co", covariant=True)

TargetMap: TypeAlias = Mapping[str, T_co]

IQArray: TypeAlias = _IQArray

ParametricWaveformDict: TypeAlias = Callable[..., TargetMap["Waveform"]]
ParametricPulseSchedule: TypeAlias = Callable[..., "PulseSchedule"]

MeasurementMode: TypeAlias = Literal["single", "avg"]
ConfigurationMode: TypeAlias = Literal["ge-ef-cr", "ge-cr-cr"]

TimeLike: TypeAlias = float | int | Time | Value
FrequencyLike: TypeAlias = float | int | Frequency | Value
