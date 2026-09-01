"""Sequencer protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy.typing as npt

from qubex.backend.quel3.interfaces.directives import DirectiveProtocol


class SequencerProtocol(Protocol):
    """Minimal sequencer protocol required by QuEL-3 execution flow."""

    def bind(
        self,
        alias: str,
        sampling_period_fs: int,
        step_samples: int,
    ) -> None:
        """Bind hardware sampling constraints for one instrument alias."""
        ...

    def register_waveform(
        self,
        name: str,
        waveform: npt.ArrayLike,
        sampling_period_ns: float | None = None,
    ) -> None:
        """Register one waveform in the sequencer library."""
        ...

    def add_event(
        self,
        instrument_alias: str,
        waveform_name: str,
        start_offset_ns: float,
        gain: float = 1.0,
        phase_offset_deg: float = 0.0,
    ) -> None:
        """Append one waveform event to the timeline."""
        ...

    def add_capture_window(
        self,
        instrument_alias: str,
        window_name: str,
        start_offset_ns: float,
        length_ns: float,
    ) -> None:
        """Append one capture window to the timeline."""
        ...

    def set_iterations(self, iterations: int) -> None:
        """Set timeline iteration count for one trigger execution."""
        ...

    def extend_length_ns(self, additional_ns: float) -> None:
        """Extend timeline length by one additional duration in ns."""
        ...

    def get_aligned_length_fs(self, post_blank_fs: int = 0) -> int:
        """Return aligned timeline length with an optional trailing blank."""
        ...

    def export_set_fixed_timeline_directive(
        self,
        instrument_alias: str,
    ) -> DirectiveProtocol:
        """Export fixed-timeline directive for one instrument alias."""
        ...


T_co = TypeVar("T_co", bound=SequencerProtocol, covariant=True)


class SequencerFactoryProtocol(Protocol[T_co]):
    """Factory protocol for quelware sequencers."""

    def __call__(
        self,
        default_sampling_period_ns: float,
        enforce_sample_grid: bool = True,
        iter_blank_ns: float = 2_000,
    ) -> T_co:
        """Create one quelware sequencer."""
        ...
