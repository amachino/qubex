"""Public API for quantum-system modeling, simulation, and pulse optimization."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import gates
from .simulation import Control, QuantumSimulator, SimulationResult, TargetUnitary
from .system import (
    ChargeBasisEigensystem,
    CompiledCosineTransmon,
    CompiledObject,
    Coupling,
    EvaluationSpace,
    GateLike,
    GateTarget,
    QuantumSystem,
    Qubit,
    Resonator,
    Transmon,
    UnitarySpecification,
)

if TYPE_CHECKING:
    from .optimization import PulseOptimizer

__all__ = [
    "ChargeBasisEigensystem",
    "CompiledCosineTransmon",
    "CompiledObject",
    "Control",
    "Coupling",
    "EvaluationSpace",
    "GateLike",
    "GateTarget",
    "PulseOptimizer",
    "QuantumSimulator",
    "QuantumSystem",
    "Qubit",
    "Resonator",
    "SimulationResult",
    "TargetUnitary",
    "Transmon",
    "UnitarySpecification",
    "gates",
]


def __getattr__(name: str) -> Any:
    """Load the deprecated pulse optimizer only when explicitly requested."""
    if name != "PulseOptimizer":
        raise AttributeError(name)
    module = importlib.import_module("qxsimulator.optimization")
    value = module.PulseOptimizer
    globals()[name] = value
    return value
