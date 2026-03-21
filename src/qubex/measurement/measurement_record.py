"""Compatibility exports for legacy measurement record imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.measurement.models.measurement_record`.
if TYPE_CHECKING:
    from .models.measurement_record import DEFAULT_RAWDATA_DIR, MeasurementRecord

    DEFAULT_DATA_DIR = DEFAULT_RAWDATA_DIR

_EXPORTS = {
    "DEFAULT_DATA_DIR": "DEFAULT_RAWDATA_DIR",
    "DEFAULT_RAWDATA_DIR": "DEFAULT_RAWDATA_DIR",
    "MeasurementRecord": "MeasurementRecord",
}

__all__ = ["DEFAULT_DATA_DIR", "DEFAULT_RAWDATA_DIR", "MeasurementRecord"]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.measurement.measurement_record",
        canonical_module="qubex.measurement.models.measurement_record",
        exports=_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=_EXPORTS)
