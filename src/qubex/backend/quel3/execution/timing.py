"""Shared sample-grid calculations for analysis and sequencer export."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from qubex.backend.quel3.models.payload import Quel3ExecutionPayload, Quel3FixedTimeline


def sample_count(time_ns: float, sampling_period_ns: float) -> int:
    """Ceil a duration to samples with the sequencer's grid tolerance."""
    samples = time_ns / sampling_period_ns
    rounded = round(samples)
    if math.isclose(samples, rounded, rel_tol=0, abs_tol=1e-3):
        return rounded
    return math.ceil(samples)


def ceil_to_grid(time_ns: float, sampling_period_fs: int) -> float:
    """Return time on the next sampling boundary, in ns."""
    period_ns = sampling_period_fs / 1e6
    return sample_count(time_ns, period_ns) * period_ns


def normalize_timeline(
    timeline: Quel3FixedTimeline,
    sampling_period_fs: int,
    waveform_durations_ns: dict[str, float],
) -> Quel3FixedTimeline:
    """Round content to samples and ensure the declared length contains it."""
    events = tuple(
        replace(
            event,
            start_offset_ns=ceil_to_grid(event.start_offset_ns, sampling_period_fs),
        )
        for event in timeline.events
    )
    windows = tuple(
        replace(
            window,
            start_offset_ns=ceil_to_grid(window.start_offset_ns, sampling_period_fs),
            length_ns=max(1, sample_count(window.length_ns, sampling_period_fs / 1e6))
            * sampling_period_fs
            / 1e6,
        )
        for window in timeline.capture_windows
    )
    end_ns = max(
        [
            timeline.length_ns,
            *(
                event.start_offset_ns + waveform_durations_ns[event.waveform_name]
                for event in events
            ),
            *(window.start_offset_ns + window.length_ns for window in windows),
        ]
    )
    return replace(timeline, events=events, capture_windows=windows, length_ns=end_ns)


def normalize_payload_timing(
    payload: Quel3ExecutionPayload,
    alias_bindings: Mapping[str, tuple[int, int]],
    default_sampling_period_ns: float,
) -> Quel3ExecutionPayload:
    """Prepare timing for standalone sequencer builds without changing input data."""
    durations = {
        name: len(waveform.iq_array)
        * (
            waveform.sampling_period_ns
            if waveform.sampling_period_ns is not None
            else default_sampling_period_ns
        )
        for name, waveform in payload.waveform_library.items()
    }
    timelines = {}
    for alias, timeline in payload.fixed_timelines.items():
        if alias not in alias_bindings:
            raise ValueError(f"Missing sequencer binding for alias: {alias}.")
        for event in timeline.events:
            if event.waveform_name not in durations:
                raise ValueError(
                    f"Unknown waveform name in event: {event.waveform_name}."
                )
        timelines[alias] = normalize_timeline(
            timeline, alias_bindings[alias][0], durations
        )
    return replace(payload, fixed_timelines=timelines)
