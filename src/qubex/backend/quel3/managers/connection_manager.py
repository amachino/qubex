"""Connection and lifecycle manager for QuEL-3 backend controller."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from qubex.backend.quel3.infra.quelware_imports import (
    import_module_with_workspace_fallback,
)
from qubex.backend.quel3.interfaces import QuelwareClientFactory
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

T = TypeVar("T")


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="quel3-connection")
    return bridge.run(factory, timeout=timeout)


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
        _run_async(self._probe_quelware_connection)
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
    def load_quelware_client_factory() -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        client_module = import_module_with_workspace_fallback("quelware_client.client")
        return client_module.create_quelware_client
