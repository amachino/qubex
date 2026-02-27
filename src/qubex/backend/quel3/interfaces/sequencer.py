"""Sequencer protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol

import numpy.typing as npt

from qubex.backend.quel3.interfaces.directives import DirectiveProtocol


class SequencerProtocol(Protocol):
    """Minimal sequencer protocol required by QuEL-3 execution flow."""

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

    def export_set_fixed_timeline_directive(
        self,
        instrument_alias: str,
        sampling_period_fs: int,
    ) -> DirectiveProtocol:
        """Export fixed-timeline directive for one instrument alias."""
        ...
