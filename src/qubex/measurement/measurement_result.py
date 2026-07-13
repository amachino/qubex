"""Compatibility exports for legacy measurement result imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.measurement.models.measure_result`.
if TYPE_CHECKING:
    from .models.measure_result import (
        MeasureData,
        MeasureMode,
        MeasureResult,
        MultipleMeasureResult,
    )

__all__ = [
    "MeasureData",
    "MeasureMode",
    "MeasureResult",
    "MultipleMeasureResult",
]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.measurement.measurement_result",
        canonical_module="qubex.measurement.models.measure_result",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
