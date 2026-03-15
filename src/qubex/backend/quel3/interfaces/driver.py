"""Instrument-driver protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

import numpy.typing as npt

from qubex.backend.quel3.interfaces.client import (
    InstrumentInfoProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.interfaces.directives import DirectiveProtocol


class IqWaveformResultProtocol(Protocol):
    """Minimal waveform result protocol."""

    @property
    def iq_array(self) -> npt.ArrayLike:
        """Return captured IQ waveform array."""
        ...


CaptureResultValues: TypeAlias = Sequence[complex] | Sequence[IqWaveformResultProtocol]


class ResultContainerProtocol(Protocol):
    """Minimal fixed-timeline result container protocol."""

    @property
    def iq_result(self) -> Mapping[str, CaptureResultValues]:
        """Return capture-window IQ results keyed by window name."""
        ...


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

    async def apply(
        self,
        directive: DirectiveProtocol | Sequence[DirectiveProtocol],
    ) -> None:
        """Apply one or more fixed-timeline directives."""
        ...

    async def initialize(self) -> None:
        """Initialize instrument state before apply/trigger flow."""
        ...

    async def fetch_result(self) -> ResultContainerProtocol:
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
