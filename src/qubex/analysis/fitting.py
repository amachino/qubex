"""Compatibility wrapper for fitting utilities migrated to qxfitting."""

from __future__ import annotations

from typing import Any

from qxfitting import fitting as _fitting

FitResult = _fitting.FitResult
FitStatus = _fitting.FitStatus


def __getattr__(name: str) -> Any:
    """Delegate attribute access to `qxfitting.fitting`."""
    return getattr(_fitting, name)


def __dir__() -> list[str]:
    """Return attributes from this module and delegated fitting module."""
    names = set(globals())
    names.update(dir(_fitting))
    return sorted(names)
