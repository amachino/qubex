"""Sequencer protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol

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

    def export_set_fixed_timeline_directive(
        self,
        instrument_alias: str,
    ) -> DirectiveProtocol:
        """Export fixed-timeline directive for one instrument alias."""
        ...
