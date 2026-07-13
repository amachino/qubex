"""Compatibility exports for legacy experiment Rabi parameter imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

# TODO: Remove this compatibility shim after downstream imports migrate to
# `qubex.experiment.models.rabi_param`.
if TYPE_CHECKING:
    from .models.rabi_param import RabiParam

__all__ = ["RabiParam"]


def __getattr__(name: str) -> Any:
    """Resolve legacy exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.experiment.rabi_param",
        canonical_module="qubex.experiment.models.rabi_param",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
