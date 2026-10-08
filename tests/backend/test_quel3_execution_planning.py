"""Behavior of QuEL-3 resource planning and result reconstruction."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend.quel3 import (
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.execution.resources import Quel3PayloadAnalyzer
from qubex.backend.quel3.instrument_cache import InstrumentCache


def make_cache(
    ports: dict[str, str] | None = None, periods: dict[str, int] | None = None
) -> InstrumentCache:
    """Build cached instruments without importing the hardware runtime."""
    cache = InstrumentCache()
    cache.replace_all(
        instrument_infos=cast(
            Any,
            [
                SimpleNamespace(
                    id=f"unit:{alias}",
                    port_id=port,
                    definition=SimpleNamespace(alias=alias),
                    config=SimpleNamespace(
                        sampling_period_fs=(periods or {}).get(alias, 800_000),
                        timeline_step_samples=4,
                    ),
                )
                for alias, port in (ports or {"R0": "unit:trx_p00p01"}).items()
            ],
        )
    )
    return cache


def make_payload(
    *,
    frequency: float | None = 6e9,
    shots: int = 5,
    samples: int = 4,
    alias: str = "R0",
    mode: Quel3CaptureMode = Quel3CaptureMode.RAW_WAVEFORMS,
) -> Quel3ExecutionPayload:
    """Build a small job with one waveform and one capture."""
    return Quel3ExecutionPayload(
        waveform_library={"shape": Quel3Waveform(np.full(samples, 0.25 + 0j), 0.8)},
        fixed_timelines={
            alias: Quel3FixedTimeline(
                events=(Quel3WaveformEvent("shape", 0, 0.5, 45),),
                capture_windows=(Quel3CaptureWindow("read", 0, samples * 0.8),),
                length_ns=samples * 0.8,
                frequency_hz=frequency,
            )
        },
        n_iterations=shots,
        shot_interval_ns=0,
        capture_mode=mode,
    )


def test_analysis_extracts_planning_metadata_and_normalizes_timelines() -> None:
    """Analysis should expose counts and conditions without requiring IQ in planning."""
    payload = make_payload()
    timeline = replace(
        payload.fixed_timelines["R0"],
        events=(Quel3WaveformEvent("shape", 0.3),),
        capture_windows=(Quel3CaptureWindow("read", 0.3, 0.3),),
        length_ns=0.1,
    )
    payload = replace(payload, fixed_timelines={"R0": timeline})
    analysis = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4).analyze_all(
        (payload,), make_cache()
    )[0]

    assert analysis.planning.requirements.waveform_samples == {"R0": 4}
    assert analysis.planning.requirements.capture_samples_per_shot == {"unit:rx_p00": 1}
    assert analysis.planning.requirements.timeline_duration_ns == pytest.approx(4.0)
    assert analysis.planning.conditions.n_iterations == 5
    assert analysis.planning.conditions.frequencies_hz == {"R0": 6e9}
    normalized = analysis.payload.fixed_timelines["R0"]
    assert normalized.events[0].start_offset_ns == pytest.approx(0.8)
    assert normalized.capture_windows[0].length_ns == pytest.approx(0.8)
    assert (
        analysis.payload.waveform_library["shape"].iq_array
        is payload.waveform_library["shape"].iq_array
    )
    assert payload.fixed_timelines["R0"] == timeline
