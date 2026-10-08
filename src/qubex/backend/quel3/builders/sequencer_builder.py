"""Build quelware sequencers from QuEL-3 execution payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import numpy as np

from qubex.backend.quel3 import quel3_backend_constants
from qubex.backend.quel3.execution.timing import normalize_payload_timing
from qubex.backend.quel3.interfaces import (
    SequencerFactoryProtocol,
    SequencerProtocol,
)
from qubex.backend.quel3.models import Quel3ExecutionPayload

T = TypeVar("T", bound=SequencerProtocol)

# Well below one LSB (~3e-5) for normalized signed 16-bit DSP amplitudes.
_WAVEFORM_AMPLITUDE_ATOL = 1e-7


class Quel3SequencerBuilder:
    """Build sequencer events and waveforms from `Quel3ExecutionPayload`."""

    def build(
        self,
        *,
        payload: Quel3ExecutionPayload,
        sequencer_factory: SequencerFactoryProtocol[T],
        default_sampling_period_ns: float,
        alias_bindings: Mapping[str, tuple[int, int]],
    ) -> T:
        """
        Build one sequencer instance from a QuEL-3 execution payload.

        Parameters
        ----------
        payload : Quel3ExecutionPayload
            QuEL-3 execution payload from measurement adapter.
        sequencer_factory : SequencerFactoryProtocol[T]
            Quelware-compatible sequencer factory.
        default_sampling_period_ns : float
            Sequencer default sampling period in ns.
        alias_bindings : Mapping[str, tuple[int, int]]
            Per-alias binding of (`sampling_period_fs`, `timeline_step_samples`).

        Returns
        -------
        T
            Built sequencer instance.

        Raises
        ------
        ValueError
            If waveform samples are nonfinite or their complex magnitude exceeds
            `1 + 1e-7`.

        Notes
        -----
        Standalone builds round offsets and capture lengths to sampling grids
        before constructing the sequencer. Backend execution uses `build_prepared`
        to forward the payload analyzer's timing without a second normalization.

        Waveforms exceeding unit magnitude by at most `1e-7` are uniformly
        scaled just below one to accommodate floating-point roundoff. Input
        waveforms and event gains are not modified.
        """
        prepared = normalize_payload_timing(
            payload, alias_bindings, default_sampling_period_ns
        )
        return self.build_prepared(
            payload=prepared,
            sequencer_factory=sequencer_factory,
            default_sampling_period_ns=default_sampling_period_ns,
            alias_bindings=alias_bindings,
        )

    def build_prepared(
        self,
        *,
        payload: Quel3ExecutionPayload,
        sequencer_factory: SequencerFactoryProtocol[T],
        default_sampling_period_ns: float,
        alias_bindings: Mapping[str, tuple[int, int]],
    ) -> T:
        """
        Forward a prepared payload to a new sequencer without rounding its timing.

        Events and captures must already lie on their alias sampling grids.
        Each timeline's declared length must contain all of its content.
        The payload analyzer and execution payload builder guarantee these
        conditions for backend execution. Standalone callers can use `build`
        to normalize an unprepared payload first.
        """
        sequencer = sequencer_factory(
            default_sampling_period_ns=default_sampling_period_ns,
        )
        sequencer.set_iterations(payload.n_iterations)

        for instrument_alias in payload.fixed_timelines:
            if instrument_alias not in alias_bindings:
                raise ValueError(
                    f"Missing sequencer binding for alias: {instrument_alias}."
                )
            sampling_period_fs, timeline_step_samples = alias_bindings[instrument_alias]
            sequencer.bind(
                instrument_alias,
                sampling_period_fs=sampling_period_fs,
                step_samples=timeline_step_samples,
            )

        for waveform_name, waveform_def in payload.waveform_library.items():
            # Convert the registered shape to quelware IQ coordinates.
            iq_array = np.conj(waveform_def.iq_array)
            if not np.all(np.isfinite(iq_array)):
                raise ValueError(
                    f"Waveform {waveform_name!r} IQ values must be finite."
                )
            peak = float(np.max(np.abs(iq_array), initial=0.0))
            if peak > 1.0 + _WAVEFORM_AMPLITUDE_ATOL:
                raise ValueError(
                    f"Waveform {waveform_name!r} peak magnitude {peak:.17g} exceeds "
                    f"1 + {_WAVEFORM_AMPLITUDE_ATOL:g}."
                )
            if peak > 1.0:
                # Leave room for rounding in complex scaling and abs evaluation.
                target_peak = 1.0 - 4 * np.finfo(np.float64).eps
                iq_array *= target_peak / peak
            sequencer.register_waveform(
                waveform_name,
                iq_array,
                sampling_period_ns=waveform_def.sampling_period_ns,
            )

        timeline_length_ns = max(
            (timeline.length_ns for timeline in payload.fixed_timelines.values()),
            default=0.0,
        )
        # Set the declared span before adding content; quelware retains its maximum.
        sequencer.extend_length_ns(timeline_length_ns + payload.shot_interval_ns)
        for instrument_alias, timeline in payload.fixed_timelines.items():
            for event in timeline.events:
                sequencer.add_event(
                    instrument_alias,
                    event.waveform_name,
                    start_offset_ns=event.start_offset_ns,
                    gain=event.gain * quel3_backend_constants.EVENT_GAIN_SCALE,
                    # Complete the conjugation for the per-event phase.
                    phase_offset_deg=-event.phase_offset_deg,
                )
            for window in timeline.capture_windows:
                sequencer.add_capture_window(
                    instrument_alias,
                    window.name,
                    start_offset_ns=window.start_offset_ns,
                    length_ns=window.length_ns,
                )
        return sequencer
