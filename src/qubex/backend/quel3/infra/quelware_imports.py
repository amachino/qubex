"""Shared helpers for importing quelware modules with workspace fallback."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _candidate_local_quelware_paths(root: Path) -> tuple[Path, ...]:
    """Return local quelware source directories checked for fallback import."""
    return (
        root / "lib" / "quelware-client-internal" / "quelware-client" / "src",
        root / "lib" / "quelware-client-internal" / "quelware-core" / "python" / "src",
        root / "packages" / "quelware-client" / "quelware-client" / "src",
        root / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
    )


def _append_local_quelware_paths() -> None:
    """Append local quelware source paths when present in the workspace."""
    root = Path(__file__).resolve().parents[5]
    for path in _candidate_local_quelware_paths(root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_module_with_workspace_fallback(module_name: str) -> ModuleType:
    """Import one module, retrying after local quelware path injection."""
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, SyntaxError):
        _append_local_quelware_paths()
        return importlib.import_module(module_name)
