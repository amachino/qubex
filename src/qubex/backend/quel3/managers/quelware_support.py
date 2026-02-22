"""Shared helpers for QuEL-3 quelware module loading and async execution."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from types import ModuleType
from typing import TypeVar

_T = TypeVar("_T")


def run_coroutine(coroutine: Coroutine[object, object, _T]) -> _T:
    """Run an async workflow from synchronous manager entrypoints."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_holder: dict[str, _T] = {}
    error_holder: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_holder["value"] = asyncio.run(coroutine)
        except BaseException as exc:
            error_holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder["value"]


def append_local_quelware_paths() -> None:
    """Append local quelware source paths when present in the workspace."""
    root = Path(__file__).resolve().parents[5]
    candidates = (
        root / "packages" / "quelware-client" / "quelware-client" / "src",
        root / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
    )
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_module_with_workspace_fallback(module_name: str) -> ModuleType:
    """Import one module, retrying after local quelware path injection."""
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, SyntaxError):
        append_local_quelware_paths()
        return importlib.import_module(module_name)
