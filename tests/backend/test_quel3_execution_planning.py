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
    Quel3ExecutionOptions,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.execution.planner import Quel3ExecutionPlanner
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


def test_planning_uses_resource_metadata_to_place_payloads_and_split_shots() -> None:
    """Planning should determine offsets and shot ranges without waveform buffers."""
    from qubex.backend.quel3.execution.resources import (
        ExecutionConditions,
        PayloadPlanningInfo,
        ResourceRequirements,
    )

    info = PayloadPlanningInfo(
        ResourceRequirements({"R0": 4}, {"unit:rx_p00": 4}, 3.2, 800_000),
        ExecutionConditions(
            5, Quel3CaptureMode.RAW_WAVEFORMS, 0.3, None, {"R0": 6e9}, 800_000
        ),
    )
    plan = Quel3ExecutionPlanner().plan(
        (info, info), options=Quel3ExecutionOptions(max_capture_samples=16)
    )
    assert plan.payload_count == 2
    assert [(run.shot_start, run.shot_stop) for run in plan.executions] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [
        (p.job_index, p.start_offset_ns) for p in plan.executions[0].placements
    ] == [(0, 0), (1, 4.0)]
    assert plan.executions[0].timeline_length_ns == pytest.approx(7.2)
    assert plan.estimated_duration_ns == pytest.approx(37.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_capture_samples", 0),
        ("max_waveform_samples", True),
        ("max_execution_duration_ns", float("inf")),
        ("merge_jobs", "yes"),
    ],
)
def test_options_reject_invalid_limits_and_types(field, value) -> None:
    """Execution options should reject invalid types and nonfinite resource limits."""
    with pytest.raises(ValueError, match=field):
        Quel3ExecutionOptions(**{field: value})
