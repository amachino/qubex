"""Connection and lifecycle manager for QuEL-3 backend controller."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Protocol

from qubex.backend.quel3.managers.quelware_support import (
    import_module_with_workspace_fallback,
    run_coroutine,
)
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
        run_coroutine(self._probe_quelware_connection())
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
        client_module = import_module_with_workspace_fallback("quelware_client.client")
        return client_module.create_quelware_client

    def set_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace target-to-instrument alias mapping."""
        self._runtime_context.set_alias_map(alias_map)

    def update_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update target-to-instrument alias mapping."""
        self._runtime_context.update_alias_map(alias_map)
