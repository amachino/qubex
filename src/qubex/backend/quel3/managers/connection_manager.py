"""Connection and lifecycle manager for QuEL-3 backend controller."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections.abc import Coroutine, Mapping
from pathlib import Path
from types import TracebackType
from typing import Protocol

from qubex.backend.quel3.quel3_runtime_context import Quel3RuntimeContext


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


class Quel3ConnectionManager:
    """Handle connect/disconnect lifecycle for QuEL-3."""

    def __init__(self, *, runtime_context: Quel3RuntimeContext) -> None:
        self._runtime_context = runtime_context

    @property
    def hash(self) -> int:
        """Return stable hash for connection-side runtime state."""
        alias_items = tuple(sorted(self._runtime_context.alias_map.items()))
        return hash(
            (
                self._runtime_context.is_connected,
                self._runtime_context.quelware_endpoint,
                self._runtime_context.quelware_port,
                self._runtime_context.trigger_wait,
                alias_items,
            )
        )

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._runtime_context.is_connected

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
        self._run_coroutine(self._probe_quelware_connection())
        self._runtime_context.set_connected(True)

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._runtime_context.set_connected(False)

    async def _probe_quelware_connection(self) -> None:
        """Probe quelware endpoint by listing resources once."""
        try:
            client_factory = self.load_quelware_client_factory()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        async with client_factory(
            self._runtime_context.quelware_endpoint,
            self._runtime_context.quelware_port,
        ) as client:
            await client.list_resource_infos()

    @staticmethod
    def load_quelware_client_factory() -> _QuelwareClientFactory:
        """Import quelware client factory lazily."""

        def _import_factory() -> _QuelwareClientFactory:
            client_module = importlib.import_module("quelware_client.client")
            return client_module.create_quelware_client

        try:
            return _import_factory()
        except (ModuleNotFoundError, SyntaxError):
            Quel3ConnectionManager.append_local_quelware_paths()
            return _import_factory()

    @staticmethod
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

    @staticmethod
    def _run_coroutine(coroutine: Coroutine[object, object, None]) -> None:
        """Run async connectivity workflow in sync controller entrypoint."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coroutine)
            return

        error_holder: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                asyncio.run(coroutine)
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_holder:
            raise error_holder["error"]

    def set_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace target-to-instrument alias mapping."""
        self._runtime_context.set_alias_map(alias_map)

    def update_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-instrument alias mapping."""
        self._runtime_context.update_alias_map(alias_map)
