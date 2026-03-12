"""Compatibility exports for legacy experiment result imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.experiment.models.experiment_result`.
if TYPE_CHECKING:
    from .models.experiment_result import (
        AmplCalibData,
        AmplRabiData,
        ExperimentResult,
        FreqRabiData,
        RabiData,
        RamseyData,
        RBData,
        SweepData,
        T1Data,
        T2Data,
        TargetData,
    )

__all__ = [
    "AmplCalibData",
    "AmplRabiData",
    "ExperimentResult",
    "FreqRabiData",
    "RBData",
    "RabiData",
    "RamseyData",
    "SweepData",
    "T1Data",
    "T2Data",
    "TargetData",
]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.experiment.experiment_result",
        canonical_module="qubex.experiment.models.experiment_result",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
