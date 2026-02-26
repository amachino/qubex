"""Connection and lifecycle manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Protocol, TypeVar

T = TypeVar("T")


class _QuelwareClient(Protocol):
    """Minimal quelware client protocol for connectivity checks."""

    async def list_resource_infos(self) -> object:
        """List available quelware resources."""
        ...


class _QuelwareClientContextManager(Protocol):
    """Async context manager protocol for quelware clients."""

    async def __aenter__(self) -> _QuelwareClient:
        """Enter the async context and return a client."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Exit the async context."""
        ...


class _QuelwareClientFactory(Protocol):
    """Factory protocol for creating quelware client contexts."""

    def __call__(self, endpoint: str, port: int) -> _QuelwareClientContextManager:
        """Create one quelware client context manager."""
        ...


def _run_coroutine(coroutine: Coroutine[object, object, T]) -> T:
    """Run an async workflow from a synchronous manager entrypoint."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_holder: dict[str, T] = {}
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


def _append_local_quelware_paths() -> None:
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


def _import_module_with_workspace_fallback(module_name: str) -> ModuleType:
    """Import one module, retrying after local quelware path injection."""
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, SyntaxError):
        _append_local_quelware_paths()
        return importlib.import_module(module_name)


class Quel3ConnectionManager:
    """Handle connect/disconnect lifecycle for QuEL-3."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
    ) -> None:
        self._is_connected = False
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port

    @property
    def hash(self) -> int:
        """Return stable hash for connection-side runtime state."""
        return hash((self._is_connected, self._quelware_endpoint, self._quelware_port))

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._is_connected

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port."""
        return self._quelware_port

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Connect backend resources for selected boxes."""
        del box_names, parallel
        if self.is_connected:
            return
        _run_coroutine(self._probe_quelware_connection())
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._is_connected = False

    async def _probe_quelware_connection(self) -> None:
        """Probe quelware endpoint by listing resources once."""
        try:
            client_factory = self.load_quelware_client_factory()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        async with client_factory(
            self._quelware_endpoint,
            self._quelware_port,
        ) as client:
            await client.list_resource_infos()

    @staticmethod
    def load_quelware_client_factory() -> _QuelwareClientFactory:
        """Import quelware client factory lazily."""
        client_module = _import_module_with_workspace_fallback("quelware_client.client")
        return client_module.create_quelware_client
