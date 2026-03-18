"""Backend hardware controller contracts and implementations."""

from __future__ import annotations

from typing import Any

from qubex.compat.deprecated_imports import (
    deprecated_module_dir,
    load_deprecated_module_attr,
)

from .backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
    BackendKind,
)

__all__ = [
    "BackendController",
    "BackendExecutionRequest",
    "BackendExecutionResult",
    "BackendKind",
]

_LEGACY_EXPORTS = ["Target", "TargetType"]


def __getattr__(name: str) -> Any:
    """Resolve deprecated backend aliases lazily."""
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    value = load_deprecated_module_attr(
        name=name,
        legacy_module="qubex.backend",
        canonical_module="qubex.system",
        exports=_LEGACY_EXPORTS,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public names exposed by this package."""
    return deprecated_module_dir(exports=[*__all__, *_LEGACY_EXPORTS])
