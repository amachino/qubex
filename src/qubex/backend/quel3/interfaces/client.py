"""Client/session protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from typing import Protocol, TypeAlias

ResourceIdProtocol: TypeAlias = str
UnitLabelProtocol: TypeAlias = str


class FixedTimelineProfileProtocol(Protocol):
    """Minimal fixed-timeline profile protocol."""

    @property
    def frequency_range_min(self) -> float | None:
        """Return lower bound of the supported frequency range."""
        ...

    @property
    def frequency_range_max(self) -> float | None:
        """Return upper bound of the supported frequency range."""
        ...


class InstrumentDefinitionProtocol(Protocol):
    """Minimal instrument-definition protocol."""

    @property
    def alias(self) -> str:
        """Return instrument alias."""
        ...

    @property
    def role(self) -> object:
        """Return instrument role enum-like value."""
        ...

    @property
    def mode(self) -> object | None:
        """Return instrument mode enum-like value when available."""
        ...

    @property
    def profile(self) -> FixedTimelineProfileProtocol | None:
        """Return instrument profile when available."""
        ...


class InstrumentModeProtocol(Protocol):
    """Instrument-mode enum value protocol."""

    @property
    def name(self) -> str:
        """Return instrument mode name."""
        ...

    @property
    def value(self) -> int:
        """Return instrument mode value."""
        ...


class InstrumentRoleProtocol(Protocol):
    """Instrument-role enum value protocol."""

    @property
    def name(self) -> str:
        """Return instrument role name."""
        ...

    @property
    def value(self) -> int:
        """Return instrument role value."""
        ...


class FixedTimelineProfileFactory(Protocol):
    """Factory protocol for fixed-timeline profile entities."""

    def __call__(
        self,
        *,
        frequency_range_min: float,
        frequency_range_max: float,
    ) -> FixedTimelineProfileProtocol:
        """Create one fixed-timeline profile."""
        ...


class InstrumentDefinitionFactory(Protocol):
    """Factory protocol for instrument-definition entities."""

    def __call__(
        self,
        *,
        alias: str,
        mode: InstrumentModeProtocol,
        role: InstrumentRoleProtocol,
        profile: FixedTimelineProfileProtocol,
    ) -> InstrumentDefinitionProtocol:
        """Create one instrument definition."""
        ...


class InstrumentModeNamespaceProtocol(Protocol):
    """Instrument-mode enum namespace protocol."""

    UNSPECIFIED: InstrumentModeProtocol
    FIXED_TIMELINE: InstrumentModeProtocol


class InstrumentRoleNamespaceProtocol(Protocol):
    """Instrument-role enum namespace protocol."""

    UNSPECIFIED: InstrumentRoleProtocol
    TRANSMITTER: InstrumentRoleProtocol
    TRANSCEIVER: InstrumentRoleProtocol
    TRANSCEIVER_LOOPBACK: InstrumentRoleProtocol
    RECEIVER: InstrumentRoleProtocol


class InstrumentInfoProtocol(Protocol):
    """Minimal instrument-info protocol."""

    @property
    def id(self) -> ResourceIdProtocol:
        """Return instrument resource identifier."""
        ...

    @property
    def port_id(self) -> ResourceIdProtocol:
        """Return instrument port identifier."""
        ...

    @property
    def definition(self) -> InstrumentDefinitionProtocol:
        """Return instrument definition."""
        ...


class ResourceCategoryProtocol(Protocol):
    """Minimal resource-category protocol."""

    @property
    def name(self) -> str:
        """Return category name."""
        ...


class PortRoleProtocol(Protocol):
    """Minimal port-role protocol."""

    @property
    def name(self) -> str:
        """Return port role name."""
        ...


class ResourceInfoProtocol(Protocol):
    """Minimal resource-info protocol."""

    @property
    def id(self) -> ResourceIdProtocol:
        """Return resource identifier."""
        ...

    @property
    def category(self) -> ResourceCategoryProtocol | str:
        """Return resource category."""
        ...


class PortInfoProtocol(Protocol):
    """Minimal port-info protocol."""

    @property
    def id(self) -> ResourceIdProtocol:
        """Return port identifier."""
        ...

    @property
    def role(self) -> PortRoleProtocol:
        """Return port role."""
        ...

    @property
    def depends_on(self) -> list[ResourceIdProtocol]:
        """Return dependent resource IDs."""
        ...


class SessionProtocol(Protocol):
    """Minimal quelware session protocol."""

    async def deploy_instruments(
        self,
        port_id: ResourceIdProtocol,
        definitions: Iterable[InstrumentDefinitionProtocol],
        append: bool = False,
    ) -> list[InstrumentInfoProtocol]:
        """Deploy one or more instruments to one port."""
        ...

    async def trigger(
        self,
        instrument_ids: Iterable[ResourceIdProtocol],
        wait_ms: int | None = None,
    ) -> int:
        """Trigger one fixed-timeline session run."""
        ...


class QuelwareClientProtocol(Protocol):
    """Minimal quelware client protocol for execution."""

    def list_unit_labels(self) -> list[UnitLabelProtocol]:
        """List available QuEL-3 unit labels."""
        ...

    async def list_resource_infos(self) -> list[ResourceInfoProtocol]:
        """List available resources."""
        ...

    async def get_instrument_info(
        self, resource_id: ResourceIdProtocol
    ) -> InstrumentInfoProtocol:
        """Get instrument info for one resource ID."""
        ...

    async def get_port_info(self, resource_id: ResourceIdProtocol) -> PortInfoProtocol:
        """Get port info for one resource ID."""
        ...

    async def dump_port_state(self, resource_id: ResourceIdProtocol) -> str:
        """Dump diagnostic state for one port resource ID."""
        ...

    def create_session(
        self,
        resource_ids: Iterable[ResourceIdProtocol],
        ttl_ms: int = 4_000,
        tentative_ttl_ms: int = 1_000,
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
