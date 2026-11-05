"""Reduce constant-envelope payloads without changing their sampled signal."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import cast

import numpy as np

from qubex.backend.quel3.models.payload import (
    Quel3CompressionHint,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)


def _candidates(length: int, quantum: int) -> set[int]:
    """Return aligned quotient-interval endpoints without scanning every length."""
    result = {quantum}
    start = quantum
    while start <= length // 2:
        end = length // (length // start)
        aligned = end // quantum * quantum
        if aligned >= quantum:
            result.add(aligned)
        start = max(start + quantum, (end // quantum + 1) * quantum)
    return result


def compress_payload(
    payload: Quel3ExecutionPayload,
    sampling_periods_fs: dict[str, int],
    max_samples: int,
) -> Quel3ExecutionPayload:
    """
    Choose shared blocks by estimated IQ and event transfer size per instrument.

    Only verified compression hints are considered. Inputs are never mutated.
    The estimate charges 16 bytes per complex sample and 32 bytes each per
    event and waveform entry, accounting conservatively for protobuf metadata.
    Search covers quotient-interval endpoints rather than all possible lengths.
    The existing planner reports infeasible memory budgets after optimization.
    """
    library = dict(payload.waveform_library)
    timelines = dict(payload.fixed_timelines)
    changed = False
    for alias, timeline in payload.fixed_timelines.items():
        names = sorted({e.waveform_name for e in timeline.events})
        regions: dict[str, tuple[int, int, int]] = {}
        candidates: set[int] = set()
        for name in names:
            waveform = library[name]
            hint = waveform.compression_hint
            period = waveform.sampling_period_ns
            if hint is None or period is None:
                continue
            a, b = hint.start_sample, hint.stop_sample
            if not (0 <= a < b <= len(waveform.iq_array)):
                continue
            if not math.isfinite(hint.detuning_ghz):
                continue
            fs = round(period * 1e6)
            quantum = sampling_periods_fs[alias] // math.gcd(
                fs, sampling_periods_fs[alias]
            )
            a = math.ceil(a / quantum) * quantum
            b = b // quantum * quantum
            if b - a < 2 * quantum:
                continue
            expected = waveform.iq_array[a] * np.exp(
                -2j * np.pi * hint.detuning_ghz * period * np.arange(b - a)
            )
            if not np.allclose(waveform.iq_array[a:b], expected, rtol=0, atol=1e-10):
                continue
            regions[name] = a, b, quantum
            candidates.update(_candidates(b - a, quantum))
        if not regions:
            continue

        def describe(
            block: int | None,
            names: list[str] = names,
            regions: dict[str, tuple[int, int, int]] = regions,
            timeline: Quel3FixedTimeline = timeline,
        ):
            pieces: dict[str, list[tuple[str, int, int, float]]] = {}
            entries: dict[str, Quel3Waveform] = {}
            for name in names:
                waveform = library[name]
                period = cast(float, waveform.sampling_period_ns)
                region = regions.get(name)
                slices = [(0, len(waveform.iq_array), 0, 0.0)]
                if block is not None and region is not None:
                    a, b, quantum = region
                    if block % quantum == 0 and (b - a) // block >= 2:
                        hint = cast(Quel3CompressionHint, waveform.compression_hint)
                        count = (b - a) // block
                        slices = [(0, a, 0, 0.0)] if a else []
                        slices.extend(
                            (
                                a,
                                a + block,
                                a + k * block,
                                math.remainder(
                                    -360 * hint.detuning_ghz * k * block * period, 360
                                ),
                            )
                            for k in range(count)
                        )
                        end = a + count * block
                        if end < len(waveform.iq_array):
                            slices.append((end, len(waveform.iq_array), end, 0.0))
                parts = []
                keys = {}
                for first, last, offset, phase in slices:
                    interval = first, last
                    key = keys.get(interval)
                    if key is None:
                        samples = waveform.iq_array[first:last]
                        digest = hashlib.blake2b(
                            samples.tobytes(), digest_size=16
                        ).hexdigest()
                        key = f"compressed_{round(period * 1e6)}_{digest}"
                        keys[interval] = key
                        entries.setdefault(key, Quel3Waveform(samples, period))
                    parts.append((key, first, offset, phase))
                pieces[name] = parts
            event_count = sum(len(pieces[e.waveform_name]) for e in timeline.events)
            stored = sum(len(w.iq_array) for w in entries.values())
            score = 16 * stored + 32 * (event_count + len(entries))
            return score, event_count, stored, pieces, entries

        baseline = describe(None)
        best = baseline
        ideal = math.sqrt(sum(b - a for a, b, _ in regions.values()) * 2)
        for block in sorted(candidates, key=lambda value: (abs(value - ideal), value)):
            minimum_events = 0
            for event in timeline.events:
                region = regions.get(event.waveform_name)
                count = (
                    (region[1] - region[0]) // block
                    if region and block % region[2] == 0
                    else 1
                )
                minimum_events += max(1, count)
            if best[2] <= max_samples and 32 * minimum_events >= best[0]:
                continue
            candidate = describe(block)
            feasible = candidate[2] <= max_samples
            best_feasible = best[2] <= max_samples
            if (feasible and not best_feasible) or (
                feasible == best_feasible and candidate[:2] < best[:2]
            ):
                best = candidate
        if best is baseline or best[0] >= baseline[0]:
            continue
        events = []
        for event in timeline.events:
            waveform = library[event.waveform_name]
            for key, _, offset, phase in best[3][event.waveform_name]:
                events.append(
                    Quel3WaveformEvent(
                        key,
                        event.start_offset_ns
                        + offset * cast(float, waveform.sampling_period_ns),
                        event.gain,
                        event.phase_offset_deg + phase,
                    )
                )
        library.update(best[4])
        timelines[alias] = replace(timeline, events=tuple(events))
        changed = True
    if not changed:
        return payload
    referenced = {e.waveform_name for t in timelines.values() for e in t.events}
    return replace(
        payload,
        waveform_library={name: library[name] for name in referenced},
        fixed_timelines=timelines,
    )
