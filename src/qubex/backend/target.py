"""Compatibility exports for legacy backend target imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

if TYPE_CHECKING:
    from qubex.system import Target, TargetType

__all__ = ["Target", "TargetType"]


def __getattr__(name: str) -> Any:
    """Resolve legacy backend target exports lazily."""
    return load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.backend.target",
        canonical_module="qubex.system",
        exports=__all__,
    )


def __dir__() -> list[str]:
    """Return the public names exposed by this compatibility shim."""
    return deprecated_module_dir(exports=__all__)
