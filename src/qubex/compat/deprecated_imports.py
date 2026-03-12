"""Helpers for temporary compatibility shims."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

_WARNED_MODULES: set[str] = set()


def _warn_deprecated_import_once(*, legacy_module: str, canonical_module: str) -> None:
    """Emit one deprecation warning per legacy module path."""
    if legacy_module in _WARNED_MODULES:
        return
    _WARNED_MODULES.add(legacy_module)
    warnings.warn(
        (
            f"`{legacy_module}` is deprecated and will be removed in a future "
            f"release. Import from `{canonical_module}` instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def load_deprecated_module_attr(
    *,
    name: str,
    legacy_module: str,
    canonical_module: str,
    exports: Sequence[str] | Mapping[str, str],
) -> Any:
    """Resolve a shim export lazily and warn when it is accessed."""
    if isinstance(exports, Mapping):
        canonical_name = exports.get(name)
    else:
        canonical_name = name if name in exports else None

    if canonical_name is None:
        raise AttributeError(f"module {legacy_module!r} has no attribute {name!r}")

    _warn_deprecated_import_once(
        legacy_module=legacy_module,
        canonical_module=canonical_module,
    )
    module = import_module(canonical_module)
    return getattr(module, canonical_name)


def deprecated_module_dir(*, exports: Sequence[str] | Mapping[str, str]) -> list[str]:
    """Return sorted public names for a compatibility shim."""
    if isinstance(exports, Mapping):
        return sorted(exports)
    return sorted(exports)


def reset_deprecated_import_warning(*, legacy_module: str) -> None:
    """Reset cached warning state for tests and repeated compatibility checks."""
    _WARNED_MODULES.discard(legacy_module)
