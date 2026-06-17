"""Instrument-driver protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

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


class ResultContainerProtocol(Protocol):
    """Minimal fixed-timeline result container protocol."""

    @property
    def iq_waveform_result(self) -> Mapping[str, Sequence[IqWaveformResultProtocol]]:
        """Return waveform capture-window IQ results keyed by window name."""
        ...

    @property
    def iq_point_result(self) -> Mapping[str, Sequence[complex]]:
        """Return point capture-window IQ results keyed by window name."""
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

    async def wait_for_result(self) -> ResultContainerProtocol:
        """Wait for one fixed-timeline execution result."""
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
