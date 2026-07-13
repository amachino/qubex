"""Core module package."""

from __future__ import annotations

from qxcore import (
    DataModel,
    Expression,
    Frequency,
    FrequencyArray,
    Model,
    MutableModel,
    Time,
    TimeArray,
    Value,
    ValueArray,
    ValueArrayLike,
    units,
)

from .async_bridge import AsyncBridge
from .parallel_executor import run_parallel, run_parallel_map
from .unit_converter import (
    normalize_frequencies_to_ghz,
    normalize_frequency_to_ghz,
    normalize_quantity,
    normalize_quantity_mapping,
    normalize_time_to_ns,
)

__all__ = [
    "AsyncBridge",
    "DataModel",
    "Expression",
    "Frequency",
    "FrequencyArray",
    "Model",
    "MutableModel",
    "Time",
    "TimeArray",
    "Value",
    "ValueArray",
    "ValueArrayLike",
    "normalize_frequencies_to_ghz",
    "normalize_frequency_to_ghz",
    "normalize_quantity",
    "normalize_quantity_mapping",
    "normalize_time_to_ns",
    "run_parallel",
    "run_parallel_map",
    "units",
]
