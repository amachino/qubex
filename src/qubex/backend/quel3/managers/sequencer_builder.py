"""Build quelware sequencers from QuEL-3 execution payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy.typing as npt

from qubex.backend.quel3.quel3_execution_payload import Quel3ExecutionPayload


class _SequencerProtocol(Protocol):
    """Minimal protocol required by the sequencer builder."""

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


T = TypeVar("T", bound=_SequencerProtocol)


class Quel3SequencerBuilder:
    """Build sequencer events and waveforms from `Quel3ExecutionPayload`."""

    def build(
        self,
        *,
        payload: Quel3ExecutionPayload,
        sequencer_factory: Callable[..., T],
        default_sampling_period_ns: float,
    ) -> T:
        """
        Build one sequencer instance from a QuEL-3 execution payload.

        Parameters
        ----------
        payload : Quel3ExecutionPayload
            QuEL-3 execution payload from measurement adapter.
        sequencer_factory : Callable[..., T]
            Sequencer class or factory compatible with quelware `Sequencer`.
        default_sampling_period_ns : float
            Sequencer default sampling period in ns.

        Returns
        -------
        T
            Built sequencer instance.
        """
        sequencer = sequencer_factory(
            default_sampling_period_ns=default_sampling_period_ns
        )

        for waveform_name, waveform_def in payload.waveform_library.items():
            sequencer.register_waveform(
                waveform_name,
                waveform_def.iq_array,
                sampling_period_ns=waveform_def.sampling_period_ns,
            )

        for instrument_alias, timeline in payload.fixed_timelines.items():
            for event in timeline.events:
                if event.waveform_name not in payload.waveform_library:
                    raise ValueError(
                        f"Unknown waveform name in event: {event.waveform_name}."
                    )
                sequencer.add_event(
                    instrument_alias,
                    event.waveform_name,
                    start_offset_ns=event.start_offset_ns,
                    gain=event.gain,
                    phase_offset_deg=event.phase_offset_deg,
                )

            for capture_window in timeline.capture_windows:
                sequencer.add_capture_window(
                    instrument_alias,
                    capture_window.name,
                    start_offset_ns=capture_window.start_offset_ns,
                    length_ns=capture_window.length_ns,
                )

        return sequencer
