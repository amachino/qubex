"""Quantum simulator package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qxsimulator import (
    ChargeBasisEigensystem,
    CompiledCosineTransmon,
    CompiledObject,
    Control,
    Coupling,
    EvaluationSpace,
    GateLike,
    GateTarget,
    QuantumSimulator,
    QuantumSystem,
    Qubit,
    Resonator,
    SimulationResult,
    TargetUnitary,
    Transmon,
    UnitarySpecification,
    gates,
)

if TYPE_CHECKING:
    from qxsimulator import PulseOptimizer

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
    module = importlib.import_module("qxsimulator")
    value = module.PulseOptimizer
    globals()[name] = value
    return value
