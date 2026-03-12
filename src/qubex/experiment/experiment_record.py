"""Compatibility exports for legacy experiment record imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.experiment.models.experiment_record`.
if TYPE_CHECKING:
    from .models.experiment_record import DEFAULT_DATA_DIR, ExperimentRecord

__all__ = ["DEFAULT_DATA_DIR", "ExperimentRecord"]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.experiment.experiment_record",
        canonical_module="qubex.experiment.models.experiment_record",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
