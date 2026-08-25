"""Public pulse-optimization API for quantum systems."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .optimization_result import OptimizationResult

if TYPE_CHECKING:
    from .pulse_optimizer import PulseOptimizer

__all__ = [
    "OptimizationResult",
    "PulseOptimizer",
]


def __getattr__(name: str) -> Any:
    """Load the deprecated pulse optimizer only when explicitly requested."""
    if name != "PulseOptimizer":
        raise AttributeError(name)
    module = importlib.import_module("qxsimulator.optimization.pulse_optimizer")
    value = module.PulseOptimizer
    globals()[name] = value
    return value
