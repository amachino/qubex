"""Compatibility exports for legacy experiment calibration note imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.experiment.models.calibration_note`.
if TYPE_CHECKING:
    from .models.calibration_note import (
        CalibrationNote,
        CrossResonanceParam,
        DragParam,
        FlatTopParam,
        Parameter,
        RabiParam,
        StateParam,
    )

__all__ = [
    "CalibrationNote",
    "CrossResonanceParam",
    "DragParam",
    "FlatTopParam",
    "Parameter",
    "RabiParam",
    "StateParam",
]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.experiment.calibration_note",
        canonical_module="qubex.experiment.models.calibration_note",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
