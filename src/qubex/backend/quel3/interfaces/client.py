"""Client/session protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from typing import Protocol


class ResourceIdProtocol(Protocol):
    """Marker protocol for quelware instrument resource IDs."""


class InstrumentDefinitionProtocol(Protocol):
    """Minimal instrument-definition protocol."""

    @property
    def role(self) -> object:
        """Return instrument role enum-like value."""
        ...


class InstrumentInfoProtocol(Protocol):
    """Minimal instrument-info protocol."""

    @property
    def port_id(self) -> object:
        """Return instrument port identifier."""
        ...

    @property
    def definition(self) -> InstrumentDefinitionProtocol:
        """Return instrument definition."""
        ...


class SessionProtocol(Protocol):
    """Minimal quelware session protocol."""

    async def trigger(
        self,
        instrument_ids: Iterable[ResourceIdProtocol],
        wait: int = 1_000_000,
    ) -> None:
        """Trigger one fixed-timeline session run."""
        ...


class QuelwareClientProtocol(Protocol):
    """Minimal quelware client protocol for execution."""

    async def list_resource_infos(self) -> object:
        """List available resources."""
        ...

    async def get_instrument_info(
        self, resource_id: ResourceIdProtocol
    ) -> InstrumentInfoProtocol:
        """Get instrument info for one resource ID."""
        ...

    def create_session(
        self,
        resource_ids: Iterable[ResourceIdProtocol],
    ) -> AbstractAsyncContextManager[SessionProtocol]:
        """Create one execution session for selected resources."""
        ...


class QuelwareClientFactory(Protocol):
    """Factory protocol for quelware clients."""

    def __call__(
        self, endpoint: str, port: int
    ) -> AbstractAsyncContextManager[QuelwareClientProtocol]:
        """Create one quelware client context manager."""
        ...
