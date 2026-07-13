"""Instrument-resolver protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol

from qubex.backend.quel3.interfaces.client import (
    InstrumentInfoProtocol,
    QuelwareClientProtocol,
    ResourceIdProtocol,
)


class InstrumentResolverProtocol(Protocol):
    """Minimal instrument-resolver protocol."""

    async def refresh(self, client: QuelwareClientProtocol) -> None:
        """Refresh instrument mapping from current resources."""
        ...

    def resolve(self, aliases: list[str]) -> list[ResourceIdProtocol]:
        """Resolve instrument aliases to resource IDs."""
        ...

    def find_inst_info_by_alias(self, alias: str) -> InstrumentInfoProtocol:
        """Return one instrument info resolved by alias."""
        ...


class InstrumentResolverFactory(Protocol):
    """Factory protocol for instrument resolver instances."""

    def __call__(self) -> InstrumentResolverProtocol:
        """Create one instrument resolver instance."""
        ...
