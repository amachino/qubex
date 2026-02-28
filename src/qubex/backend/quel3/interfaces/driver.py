"""Instrument-driver protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol

from qubex.backend.quel3.interfaces.client import (
    InstrumentInfoProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.interfaces.directives import DirectiveProtocol


class InstrumentConfigProtocol(Protocol):
    """Minimal instrument-config protocol."""

    @property
    def sampling_period_fs(self) -> int:
        """Return sampling period in femtoseconds."""
        ...

    @property
    def timeline_step_samples(self) -> int:
        """Return timeline-step alignment size in samples."""
        ...


class InstrumentDriverProtocol(Protocol):
    """Minimal instrument driver protocol used in execution."""

    @property
    def instrument_config(self) -> InstrumentConfigProtocol:
        """Return instrument runtime configuration."""
        ...

    async def apply(self, directive: DirectiveProtocol) -> None:
        """Apply one fixed-timeline directive."""
        ...

    async def initialize(self) -> None:
        """Initialize instrument state before apply/trigger flow."""
        ...

    async def fetch_result(self) -> object:
        """Fetch one fixed-timeline execution result."""
        ...


class InstrumentDriverFactory(Protocol):
    """Factory protocol for fixed-timeline instrument drivers."""

    def __call__(
        self,
        session: SessionProtocol,
        instrument_info: InstrumentInfoProtocol,
    ) -> InstrumentDriverProtocol:
        """Create one instrument driver for a resource."""
        ...
