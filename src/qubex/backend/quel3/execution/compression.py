"""Reduce constant-envelope payloads without changing their sampled signal."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from qubex.backend.quel3.models.payload import (
    Quel3ExecutionPayload,
    Quel3Waveform,
    Quel3WaveformEvent,
)


@dataclass(frozen=True)
class _Region:
    """Verified constant envelope on an instrument's sample grid."""

    start: int
    stop: int
    quantum: int
    detuning_ghz: float


@dataclass(frozen=True)
class _Split:
    """A prefix, repeated block, and combined remainder and suffix."""

    start: int
    block: int
    repeats: int

    @property
    def tail_start(self) -> int:
        """Return the first sample after the repeated blocks."""
        return self.start + self.block * self.repeats


def _verify_region(waveform: Quel3Waveform, sampling_fs: int) -> _Region | None:
    """Check the hint against the actual IQ before selecting any split."""
    hint = waveform.compression_hint
    period = waveform.sampling_period_ns
    if hint is None or period is None:
        return None
    start, stop = hint.start_sample, hint.stop_sample
    if not (0 <= start < stop <= len(waveform.iq_array)):
        return None
    if not math.isfinite(hint.detuning_ghz):
        return None
    quantum = sampling_fs // math.gcd(round(period * 1e6), sampling_fs)
    start = (start + quantum - 1) // quantum * quantum
    stop = stop // quantum * quantum
    if stop - start < 2 * quantum:
        return None
    expected = waveform.iq_array[start] * np.exp(
        -2j * np.pi * hint.detuning_ghz * period * np.arange(stop - start)
    )
    if not np.allclose(waveform.iq_array[start:stop], expected, rtol=0, atol=1e-10):
        return None
    return _Region(start, stop, quantum, hint.detuning_ghz)


def _choose_split(
    length: int, region: _Region, target: int, minimum: int
) -> _Split | None:
    """Choose near-target repeats arithmetically while protecting every fragment."""
    start, stop, quantum = region.start, region.stop, region.quantum
    if 0 < start < minimum:
        start = (minimum + quantum - 1) // quantum * quantum
    if 0 < length - stop < minimum:
        stop = (length - minimum) // quantum * quantum
    span = stop - start
    # Round the repeat count to the nearest integer, resolving ties upward.
    repeats = (2 * span + target) // (2 * target)
    if repeats < 2:
        return None
    block = span // (repeats * quantum) * quantum
    tail = length - start - repeats * block
    if 0 < tail < minimum:
        block = (length - start - minimum) // (repeats * quantum) * quantum
        tail = length - start - repeats * block
    if block < minimum or (0 < tail < minimum):
        return None
    # The shared block is stored once; prefix and tail are stored separately.
    if start + block + tail >= length:
        return None
    return _Split(start, block, repeats)


def _build_events(
    waveform: Quel3Waveform,
    event: Quel3WaveformEvent,
    region: _Region,
    split: _Split,
    library: dict[str, Quel3Waveform],
) -> list[Quel3WaveformEvent]:
    """Materialize only the selected pieces and correct repeated-block phases."""
    period = cast(float, waveform.sampling_period_ns)
    pieces = [
        (0, split.start),
        (split.start, split.start + split.block),
        (split.tail_start, len(waveform.iq_array)),
    ]
    names = []
    for first, last in pieces:
        if first == last:
            names.append(None)
            continue
        samples = waveform.iq_array[first:last]
        digest = hashlib.blake2b(samples.tobytes(), digest_size=16).hexdigest()
        name = f"compressed_{round(period * 1e6)}_{digest}"
        library.setdefault(name, Quel3Waveform(samples, period))
        names.append(name)
    result = []
    prefix_name, shared_name, tail_name = names
    if prefix_name is not None:
        result.append(replace(event, waveform_name=prefix_name))
    block_name = cast(str, shared_name)
    for index in range(split.repeats):
        offset = split.start + index * split.block
        phase = math.remainder(
            -360 * region.detuning_ghz * index * split.block * period, 360
        )
        result.append(
            replace(
                event,
                waveform_name=block_name,
                start_offset_ns=event.start_offset_ns + offset * period,
                phase_offset_deg=event.phase_offset_deg + phase,
            )
        )
    if tail_name is not None:
        result.append(
            replace(
                event,
                waveform_name=tail_name,
                start_offset_ns=event.start_offset_ns + split.tail_start * period,
            )
        )
    return result


def compress_payload(
    payload: Quel3ExecutionPayload,
    sampling_periods_fs: dict[str, int],
    *,
    target_block_samples: int = 5000,
    min_chunk_samples: int = 1000,
) -> Quel3ExecutionPayload:
    """
    Share near-target blocks without creating fragments below the minimum length.

    Counts use registered waveform samples, after readout normalization. Short
    original waveforms remain unchanged. Prefixes and tails include adjacent
    constant-envelope samples when needed to meet the minimum. Repeated blocks
    preserve detuning through per-event phase correction. Inputs are not mutated.
    Memory feasibility remains the execution planner's responsibility.
    """
    if not 0 < min_chunk_samples <= target_block_samples:
        raise ValueError("Compression lengths must satisfy 0 < minimum <= target.")
    library = dict(payload.waveform_library)
    timelines = dict(payload.fixed_timelines)
    changed = False
    for alias, timeline in payload.fixed_timelines.items():
        replacements: dict[str, tuple[_Region, _Split]] = {}
        for name in sorted({event.waveform_name for event in timeline.events}):
            waveform = payload.waveform_library[name]
            region = _verify_region(waveform, sampling_periods_fs[alias])
            if region is None:
                continue
            split = _choose_split(
                len(waveform.iq_array), region, target_block_samples, min_chunk_samples
            )
            if split is not None:
                replacements[name] = region, split
        if not replacements:
            continue
        templates = {
            name: _build_events(
                payload.waveform_library[name],
                Quel3WaveformEvent(name, 0),
                region,
                split,
                library,
            )
            for name, (region, split) in replacements.items()
        }
        events = []
        for event in timeline.events:
            parts = templates.get(event.waveform_name)
            if parts is None:
                events.append(event)
                continue
            events.extend(
                replace(
                    part,
                    start_offset_ns=event.start_offset_ns + part.start_offset_ns,
                    gain=event.gain,
                    phase_offset_deg=event.phase_offset_deg + part.phase_offset_deg,
                )
                for part in parts
            )
        timelines[alias] = replace(timeline, events=tuple(events))
        changed = True
    if not changed:
        return payload
    referenced = {
        event.waveform_name
        for timeline in timelines.values()
        for event in timeline.events
    }
    return replace(
        payload,
        waveform_library={name: library[name] for name in referenced},
        fixed_timelines=timelines,
    )
