"""Runtime context shared across QuEL-3 backend managers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class Quel3RuntimeContextReader(Protocol):
    """Read-only interface for QuEL-3 runtime state shared by managers."""

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        ...

    @property
    def alias_map(self) -> Mapping[str, str]:
        """Return target-to-instrument alias map."""
        ...

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware API endpoint."""
        ...

    @property
    def quelware_port(self) -> int:
        """Return quelware API port."""
        ...

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        ...


class Quel3RuntimeContext:
    """Mutable runtime state shared by QuEL-3 backend managers."""

    def __init__(
        self,
        *,
        alias_map: Mapping[str, str],
        quelware_endpoint: str,
        quelware_port: int,
        sampling_period: float,
    ) -> None:
        self._is_connected = False
        self._alias_map: dict[str, str] = dict(alias_map)
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._sampling_period = sampling_period

    @property
    def is_connected(self) -> bool:
        """Return whether runtime resources are connected."""
        return self._is_connected

    @property
    def alias_map(self) -> Mapping[str, str]:
        """Return target-to-instrument alias map."""
        return self._alias_map

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware API endpoint."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware API port."""
        return self._quelware_port

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        return self._sampling_period

    def set_connected(self, connected: bool) -> None:
        """Update connected state."""
        self._is_connected = connected

    def set_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Replace full alias map."""
        self._alias_map = dict(alias_map)

    def update_alias_map(self, alias_map: Mapping[str, str]) -> None:
        """Update alias map entries."""
        self._alias_map.update(alias_map)
