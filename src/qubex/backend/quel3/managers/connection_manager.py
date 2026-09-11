"""Connection and lifecycle manager for QuEL-3 backend controller."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.interfaces import QuelwareClientFactory
from qubex.backend.quel3.interfaces.client import UnitLabelProtocol
from qubex.backend.quel3.managers.runtime_config import Quel3RuntimeConfig
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
        runtime_config: Quel3RuntimeConfig | None = None,
    ) -> None:
        self._is_connected = False
        self._runtime_config = runtime_config or Quel3RuntimeConfig()

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return the shared quelware runtime config."""
        return self._runtime_config

    @property
    def hash(self) -> int:
        """Return stable hash for connection-side runtime state."""
        return hash((self._is_connected, self._runtime_config))

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._is_connected

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int | None:
        """Return quelware port."""
        return self._runtime_config.port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._runtime_config.client_mode_value

    @property
    def quelware_pat_path(self) -> str | None:
        """Return configured quelware personal access token path."""
        return self._runtime_config.pat_path

    def connect(
        self,
        unit_labels: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect after verifying that all selected unit labels exist.

        `None` or an empty list only probes endpoint availability. Each call
        rechecks discovery, including when already connected. Failure leaves
        the manager disconnected.

        Raises
        ------
        ValueError
            A selected unit label is absent from the endpoint.
        """
        del parallel
        self._is_connected = False
        selected_unit_labels = (
            [unit_labels] if isinstance(unit_labels, str) else unit_labels or []
        )
        found_unit_labels = _run_async(self._probe_quelware_connection)
        missing_unit_labels = set(selected_unit_labels) - set(found_unit_labels)
        if missing_unit_labels:
            raise ValueError(
                f"QuEL-3 units were not discovered: {sorted(missing_unit_labels)}. "
                f"Available unit labels: {sorted(found_unit_labels)}."
            )
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        self._is_connected = False

    async def _probe_quelware_connection(self) -> list[UnitLabelProtocol]:
        """Probe quelware endpoint by listing units once."""
        try:
            client_factory = self.load_quelware_client_factory()
        except (ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "quelware-client is not available. Install compatible quelware packages or configure PYTHONPATH."
            ) from exc

        async with client_factory(
            self._runtime_config.endpoint,
            self._runtime_config.port,
        ) as client:
            return client.list_unit_labels()

    def load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()
