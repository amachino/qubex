"""Build quelware sequencers from QuEL-3 execution payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeVar

from qubex.backend.quel3.interfaces import (
    SequencerFactoryProtocol,
    SequencerProtocol,
)
from qubex.backend.quel3.models import Quel3ExecutionPayload

T = TypeVar("T", bound=SequencerProtocol)

_QUEL3_CLOCK_FREQUENCY_HZ = 312_500_000
_TRIGGER_GRID_TICKS = 32
_TRIGGER_GRID_NS = _TRIGGER_GRID_TICKS * (1e9 / _QUEL3_CLOCK_FREQUENCY_HZ)
_MIN_SHOT_INTERVAL_NS = 1_024.0
_TIME_GRID_SAMPLE_ATOL = 1e-3


class Quel3SequencerBuilder:
    """Build sequencer events and waveforms from `Quel3ExecutionPayload`."""

    @staticmethod
    def _resolve_effective_shot_interval_ns(shot_interval_ns: float) -> float:
        effective_shot_interval_ns = max(shot_interval_ns, _MIN_SHOT_INTERVAL_NS)
        return (
            math.ceil(effective_shot_interval_ns / _TRIGGER_GRID_NS) * _TRIGGER_GRID_NS
        )

    @staticmethod
    def _ceil_to_sampling_grid_ns(time_ns: float, sampling_period_fs: int) -> float:
        """Return time ceiled to the alias sampling grid in ns."""
        sampling_period_ns = sampling_period_fs / 1e6
        samples = time_ns / sampling_period_ns
        rounded_samples = round(samples)
        if math.isclose(
            samples,
            rounded_samples,
            rel_tol=0.0,
            abs_tol=_TIME_GRID_SAMPLE_ATOL,
        ):
            return float(rounded_samples) * sampling_period_ns
        return float(math.ceil(samples)) * sampling_period_ns

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
        """
        iter_blank_ns = (
            self._resolve_effective_shot_interval_ns(payload.shot_interval_ns)
            if payload.shot_interval_ns > 0
            else 0.0
        )
        sequencer = sequencer_factory(
            default_sampling_period_ns=default_sampling_period_ns,
            iter_blank_ns=iter_blank_ns,
        )
        sequencer.set_iterations(payload.n_iterations)

        for instrument_alias in payload.fixed_timelines:
            binding = alias_bindings.get(instrument_alias)
            if binding is None:
                raise ValueError(
                    f"Missing sequencer binding for alias: {instrument_alias}."
                )
            sampling_period_fs, timeline_step_samples = binding
            sequencer.bind(
                instrument_alias,
                sampling_period_fs=sampling_period_fs,
                step_samples=timeline_step_samples,
            )

        for waveform_name, waveform_def in payload.waveform_library.items():
            sequencer.register_waveform(
                waveform_name,
                waveform_def.iq_array,
                sampling_period_ns=waveform_def.sampling_period_ns,
            )

        for instrument_alias, timeline in payload.fixed_timelines.items():
            sampling_period_fs = alias_bindings[instrument_alias][0]
            for event in timeline.events:
                if event.waveform_name not in payload.waveform_library:
                    raise ValueError(
                        f"Unknown waveform name in event: {event.waveform_name}."
                    )
                sequencer.add_event(
                    instrument_alias,
                    event.waveform_name,
                    start_offset_ns=self._ceil_to_sampling_grid_ns(
                        event.start_offset_ns,
                        sampling_period_fs,
                    ),
                    gain=event.gain,
                    # Invert sign
                    phase_offset_deg=-event.phase_offset_deg,
                )

            for capture_window in timeline.capture_windows:
                sequencer.add_capture_window(
                    instrument_alias,
                    capture_window.name,
                    start_offset_ns=self._ceil_to_sampling_grid_ns(
                        capture_window.start_offset_ns,
                        sampling_period_fs,
                    ),
                    length_ns=self._ceil_to_sampling_grid_ns(
                        capture_window.length_ns,
                        sampling_period_fs,
                    ),
                )

        return sequencer
