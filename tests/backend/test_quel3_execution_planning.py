"""Behavior of QuEL-3 resource planning and result reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionOptions,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.execution.payload_builder import (
    BuiltExecution,
    Quel3ExecutionPayloadBuilder,
)
from qubex.backend.quel3.execution.planner import Quel3ExecutionPlanner
from qubex.backend.quel3.execution.resources import Quel3PayloadAnalyzer
from qubex.backend.quel3.execution.results import Quel3ResultAssembler
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.managers.execution_manager import Quel3ExecutionManager


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


@dataclass(frozen=True)
class _PreparedExecutions:
    jobs: tuple[Quel3ExecutionPayload, ...]
    executions: tuple[BuiltExecution, ...]
    estimated_duration_ns: float


def plan_jobs(
    *payloads: Quel3ExecutionPayload,
    cache: InstrumentCache | None = None,
    **options: Any,
) -> _PreparedExecutions:
    """Analyze, plan, and build payloads for end-to-end packing assertions."""
    analyses = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4).analyze_all(
        payloads, cache or make_cache()
    )
    plan = Quel3ExecutionPlanner().plan(
        tuple(analysis.planning for analysis in analyses),
        options=Quel3ExecutionOptions(**options),
    )
    return _PreparedExecutions(
        tuple(analysis.payload for analysis in analyses),
        Quel3ExecutionPayloadBuilder().build_all(analyses, plan),
        plan.estimated_duration_ns,
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


def test_planning_uses_resource_metadata_to_place_whole_payloads() -> None:
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
        (info, info), options=Quel3ExecutionOptions(max_capture_samples=40)
    )
    assert plan.payload_count == 2
    assert [(run.shot_start, run.shot_stop) for run in plan.executions] == [
        (0, 5),
    ]
    assert [
        (p.job_index, p.start_offset_ns) for p in plan.executions[0].placements
    ] == [(0, 0), (1, 4.0)]
    assert plan.executions[0].timeline_length_ns == pytest.approx(7.2)
    assert plan.estimated_duration_ns == pytest.approx(37.5)


def test_payload_builder_applies_supplied_placements_and_maps_captures() -> None:
    """Payload construction should use plan offsets and retain each capture identity."""
    from qubex.backend.quel3 import (
        Quel3ExecutionPlan,
        Quel3PayloadPlacement,
        Quel3PlannedExecution,
        Quel3ResourceUsage,
    )

    analyses = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4).analyze_all(
        (make_payload(), make_payload()), make_cache()
    )
    run = Quel3PlannedExecution(
        (Quel3PayloadPlacement(0, 0), Quel3PayloadPlacement(1, 10.4)),
        1,
        3,
        13.6,
        Quel3ResourceUsage({"R0": 8}, {"unit:rx_p00": 16}, 27.2),
    )
    built = Quel3ExecutionPayloadBuilder().build_all(
        analyses, Quel3ExecutionPlan(2, (run,))
    )[0]
    timeline = built.payload.fixed_timelines["R0"]
    assert [event.start_offset_ns for event in timeline.events] == [0, 10.4]
    assert timeline.length_ns == pytest.approx(13.6)
    assert built.payload.n_iterations == 2
    assert built.result_mapping.capture_counts == {"R0": 2}
    assert [
        (fragment.job_index, fragment.shot_start, fragment.shot_stop)
        for fragment in built.result_mapping.fragments
    ] == [(0, 1, 3), (1, 1, 3)]
    assert [
        fragment.captures[0].execution_index
        for fragment in built.result_mapping.fragments
    ] == [0, 1]


@pytest.mark.parametrize(
    "mode", [mode for mode in Quel3CaptureMode if mode != Quel3CaptureMode.UNSPECIFIED]
)
def test_capture_limit_splits_all_modes_using_raw_samples(mode) -> None:
    """RAW capacity should split every capture mode into the same shot ranges."""
    plan = plan_jobs(make_payload(mode=mode), max_capture_samples=8)
    assert [run.payload.n_iterations for run in plan.executions] == [2, 2, 1]
    assert [
        (
            run.result_mapping.fragments[0].shot_start,
            run.result_mapping.fragments[0].shot_stop,
        )
        for run in plan.executions
    ] == [(0, 2), (2, 4), (4, 5)]
    assert [run.resources.capture_samples for run in plan.executions] == [
        {"unit:rx_p00": 8},
        {"unit:rx_p00": 8},
        {"unit:rx_p00": 4},
    ]


@pytest.mark.parametrize(("limit", "counts"), [(19, [4, 1]), (20, [5]), (21, [5])])
def test_capture_limit_is_inclusive(limit, counts) -> None:
    """Capture capacity should accept the exact limit and split just above it."""
    assert [
        run.payload.n_iterations
        for run in plan_jobs(make_payload(), max_capture_samples=limit).executions
    ] == counts


def test_capture_capacity_is_summed_across_instruments_on_one_mux() -> None:
    """Instruments sharing a receiver port should consume one combined budget."""
    first = make_payload()
    second = make_payload(alias="R1")
    payload = replace(
        first, fixed_timelines={**first.fixed_timelines, **second.fixed_timelines}
    )
    cache = make_cache({"R0": "unit:trx_p00p01", "R1": "unit:trx_p00p02"})
    assert [
        run.payload.n_iterations
        for run in plan_jobs(payload, cache=cache, max_capture_samples=16).executions
    ] == [2, 2, 1]


def test_distinct_units_have_independent_capture_budgets() -> None:
    """Identical receiver numbers on different units should have separate budgets."""
    payload = make_payload()
    other = make_payload(alias="R1")
    payload = replace(
        payload, fixed_timelines={**payload.fixed_timelines, **other.fixed_timelines}
    )
    cache = make_cache({"R0": "unit:trx_p00p01", "R1": "other:trx_p00p01"})
    assert len(plan_jobs(payload, cache=cache, max_capture_samples=20).executions) == 1


def test_frequency_changes_preserve_input_order() -> None:
    """A B A jobs should execute in input order as three separate runs."""
    plan = plan_jobs(
        make_payload(),
        make_payload(frequency=6.1e9),
        make_payload(),
    )
    assert [run.result_mapping.fragments[0].job_index for run in plan.executions] == [
        0,
        1,
        2,
    ]
    assert [
        run.payload.fixed_timelines["R0"].frequency_hz for run in plan.executions
    ] == [6e9, 6.1e9, 6e9]


def test_unspecified_frequency_keeps_jobs_separate() -> None:
    """An inherited hardware frequency should execute in its own run."""
    plan = plan_jobs(
        make_payload(),
        make_payload(frequency=None),
        make_payload(),
    )
    assert [run.result_mapping.fragments[0].job_index for run in plan.executions] == [
        0,
        1,
        2,
    ]


def test_packing_checks_frequencies_only_against_current_run() -> None:
    """A completed run's frequency should not prevent packing later adjacent jobs."""
    plan = plan_jobs(
        make_payload(shots=2),
        make_payload(alias="R1", shots=1, frequency=5e9),
        make_payload(shots=1, frequency=6.1e9),
        cache=make_cache({"R0": "unit:trx_p00p01", "R1": "unit:trx_p02p03"}),
    )
    assert [
        tuple(fragment.job_index for fragment in run.result_mapping.fragments)
        for run in plan.executions
    ] == [(0,), (1, 2)]
    assert plan.executions[1].payload.fixed_timelines["R0"].frequency_hz == 6.1e9


def test_merging_keeps_job_waveforms_and_preserves_events() -> None:
    """Merging should namespace each job's waveforms and preserve gain and phase."""
    first, second = make_payload(), make_payload()
    plan = plan_jobs(first, second, max_waveform_samples=8)
    assert len(plan.executions) == 1
    run = plan.executions[0]
    assert run.resources.waveform_samples == {"R0": 8}
    assert len(run.payload.waveform_library) == 2
    events = run.payload.fixed_timelines["R0"].events
    assert [(e.start_offset_ns, e.gain, e.phase_offset_deg) for e in events] == [
        (0, 0.5, 45),
        (3.2, 0.5, 45),
    ]
    assert (
        next(iter(run.payload.waveform_library.values())).iq_array
        is first.waveform_library["shape"].iq_array
    )
    assert len(first.fixed_timelines["R0"].events) == 1
    assert events[0].waveform_name != events[1].waveform_name
    assert (
        run.payload.waveform_library[events[1].waveform_name].iq_array
        is second.waveform_library["shape"].iq_array
    )


@pytest.mark.parametrize(
    "mode", [mode for mode in Quel3CaptureMode if mode != Quel3CaptureMode.UNSPECIFIED]
)
@pytest.mark.parametrize(
    ("options", "groups"),
    [
        ({"max_capture_samples": 40}, [(0, 1), (2,)]),
        ({"max_capture_samples": 39}, [(0,), (1,), (2,)]),
        ({"max_execution_duration_ns": 37.5}, [(0, 1), (2,)]),
        ({"max_execution_duration_ns": 37.49}, [(0,), (1,), (2,)]),
    ],
)
def test_packing_preserves_all_shots_at_capture_and_duration_limits(
    mode, options, groups
) -> None:
    """Packing should start a new group when all shots would exceed a resource limit."""
    payload = replace(make_payload(mode=mode), shot_interval_ns=0.3)
    plan = plan_jobs(payload, payload, payload, **options)
    assert [
        tuple(fragment.job_index for fragment in run.result_mapping.fragments)
        for run in plan.executions
    ] == groups
    assert [run.payload.n_iterations for run in plan.executions] == [5] * len(groups)
    assert all(
        (fragment.shot_start, fragment.shot_stop) == (0, 5)
        for run in plan.executions
        for fragment in run.result_mapping.fragments
    )


@pytest.mark.parametrize(
    "options", [{"max_capture_samples": 20}, {"max_execution_duration_ns": 16}]
)
def test_only_oversized_single_jobs_are_split_between_whole_groups(options) -> None:
    """An oversized job should split alone while neighboring small jobs pack whole."""
    small = make_payload(samples=1)
    plan = plan_jobs(small, small, make_payload(samples=8), small, small, **options)
    assert [
        tuple(fragment.job_index for fragment in run.result_mapping.fragments)
        for run in plan.executions
    ] == [(0, 1), (2,), (2,), (2,), (3, 4)]
    assert [run.payload.n_iterations for run in plan.executions] == [5, 2, 2, 1, 5]
    assert [
        (
            run.result_mapping.fragments[0].shot_start,
            run.result_mapping.fragments[0].shot_stop,
        )
        for run in plan.executions
    ] == [(0, 5), (0, 2), (2, 4), (4, 5), (0, 5)]


def test_adjacent_oversized_jobs_are_split_independently() -> None:
    """Two oversized neighbors should finish their own shots in input order."""
    plan = plan_jobs(make_payload(), make_payload(), max_capture_samples=16)
    assert [
        (fragment.job_index, fragment.shot_start, fragment.shot_stop)
        for run in plan.executions
        for fragment in run.result_mapping.fragments
    ] == [(0, 0, 4), (0, 4, 5), (1, 0, 4), (1, 4, 5)]


def test_packing_starts_new_execution_at_waveform_limit() -> None:
    """Packing should fill consecutive jobs up to capacity before starting a new run."""
    plan = plan_jobs(
        make_payload(), make_payload(), make_payload(), max_waveform_samples=8
    )
    assert [
        tuple(fragment.job_index for fragment in run.result_mapping.fragments)
        for run in plan.executions
    ] == [(0, 1), (2,)]
    assert [run.resources.waveform_samples for run in plan.executions] == [
        {"R0": 8},
        {"R0": 4},
    ]


def test_equal_waveforms_in_different_jobs_consume_separate_budget() -> None:
    """Identical IQ samples in different jobs should count as separate waveforms."""
    plan = plan_jobs(make_payload(), make_payload(), max_waveform_samples=4)
    assert len(plan.executions) == 2
    assert [run.result_mapping.fragments[0].job_index for run in plan.executions] == [
        0,
        1,
    ]


def test_merging_preserves_different_waveforms_with_the_same_name() -> None:
    """Job-local waveform names should retain their own IQ samples after packing."""
    first = make_payload()
    second = replace(
        first,
        waveform_library={"shape": Quel3Waveform(np.full(4, 0.5 + 0j), 0.8)},
    )
    run = plan_jobs(first, second).executions[0]
    events = run.payload.fixed_timelines["R0"].events
    for event, expected in zip(events, (0.25, 0.5), strict=True):
        np.testing.assert_array_equal(
            run.payload.waveform_library[event.waveform_name].iq_array,
            np.full(4, expected, dtype=np.complex128),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"n_iterations": 3},
        {"shot_interval_ns": 0.8},
        {"capture_mode": Quel3CaptureMode.AVERAGED_VALUE},
    ],
)
def test_jobs_with_different_execution_settings_remain_separate(changes) -> None:
    """Packing should preserve each job's iterations, interval, and capture mode."""
    first = make_payload()
    second = replace(first, **changes)
    plan = plan_jobs(first, second)
    assert len(plan.executions) == 2
    assert [run.result_mapping.fragments[0].job_index for run in plan.executions] == [
        0,
        1,
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"max_waveform_samples": 3},
        {"max_capture_samples": 3},
        {"max_execution_duration_ns": 3},
    ],
)
def test_unsplittable_job_reports_the_violated_resource(options) -> None:
    """A job exceeding a one-shot resource limit should fail before execution."""
    with pytest.raises(ValueError, match=r"job 0.*limit"):
        plan_jobs(make_payload(), **options)


def test_time_limit_splits_even_when_merging_is_disabled() -> None:
    """Disabling job merging should retain automatic duration-based splitting."""
    plan = plan_jobs(make_payload(), merge_jobs=False, max_execution_duration_ns=6.4)
    assert [run.payload.n_iterations for run in plan.executions] == [2, 2, 1]


@pytest.mark.parametrize(
    "mode", [mode for mode in Quel3CaptureMode if mode != Quel3CaptureMode.UNSPECIFIED]
)
def test_result_assembly_preserves_shots_and_weights_unequal_chunks(mode) -> None:
    """Reconstruction should concatenate shots or weight chunk averages by shots."""
    plan = plan_jobs(make_payload(mode=mode), max_capture_samples=8)
    assembler = Quel3ResultAssembler(plan.jobs)
    is_waveform = mode in (
        Quel3CaptureMode.RAW_WAVEFORMS,
        Quel3CaptureMode.AVERAGED_WAVEFORM,
    )
    is_average = mode in (
        Quel3CaptureMode.AVERAGED_VALUE,
        Quel3CaptureMode.AVERAGED_WAVEFORM,
    )
    for run in plan.executions:
        fragment = run.result_mapping.fragments[0]
        values = np.arange(fragment.shot_start, fragment.shot_stop, dtype=np.complex128)
        if is_waveform:
            values = np.repeat(values[:, None], 4, axis=1)
        if is_average:
            values = np.atleast_1d(values.mean(axis=0))
        assembler.add(
            run.result_mapping,
            Quel3BackendExecutionResult(
                {}, {"R0": [values]}, {"sampling_period_ns": 0.8}
            ),
        )
    result = assembler.finish()[0]
    if is_average:
        np.testing.assert_allclose(result.data["R0"][0], 2, rtol=0, atol=1e-12)
    else:
        assert result.data["R0"][0].shape == ((5, 4) if is_waveform else (5,))
        np.testing.assert_allclose(
            result.data["R0"][0].reshape(5, -1)[:, 0], np.arange(5), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "mode", [mode for mode in Quel3CaptureMode if mode != Quel3CaptureMode.UNSPECIFIED]
)
def test_split_and_whole_jobs_restore_each_capture_sequence(mode) -> None:
    """Split and whole jobs should restore every capture in timeline order."""
    first = make_payload(mode=mode)
    timeline = first.fixed_timelines["R0"]
    first = replace(
        first,
        fixed_timelines={
            "R0": replace(
                timeline,
                capture_windows=(
                    Quel3CaptureWindow("late", 3.2, 3.2),
                    *timeline.capture_windows,
                ),
            )
        },
    )
    plan = plan_jobs(first, make_payload(mode=mode), max_capture_samples=24)
    assert [run.payload.n_iterations for run in plan.executions] == [3, 2, 5]
    assembler = Quel3ResultAssembler(plan.jobs)
    waveform = mode in (
        Quel3CaptureMode.RAW_WAVEFORMS,
        Quel3CaptureMode.AVERAGED_WAVEFORM,
    )
    averaged = mode in (
        Quel3CaptureMode.AVERAGED_WAVEFORM,
        Quel3CaptureMode.AVERAGED_VALUE,
    )
    for run in plan.executions:
        fragment = run.result_mapping.fragments[0]
        start, stop = fragment.shot_start, fragment.shot_stop
        captures = []
        for capture in fragment.captures:
            offset = fragment.job_index * 20 + capture.original_index * 10
            values = np.arange(start, stop, dtype=np.complex128) + offset
            if waveform:
                values = np.repeat(values[:, None], 4, axis=1)
            if averaged:
                values = np.atleast_1d(values.mean(axis=0))
            captures.append(values)
        assembler.add(
            run.result_mapping,
            Quel3BackendExecutionResult(
                {}, {"R0": captures}, {"sampling_period_ns": 0.8}
            ),
        )
    results = assembler.finish()
    assert [len(result.data["R0"]) for result in results] == [2, 1]
    for capture, offset in zip(
        [*results[0].data["R0"], *results[1].data["R0"]], (0, 10, 20), strict=True
    ):
        if averaged:
            np.testing.assert_allclose(capture, offset + 2, rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose(
                capture.reshape(5, -1)[:, 0], np.arange(5) + offset, rtol=0, atol=0
            )


def test_result_assembly_rejects_missing_shots() -> None:
    """Incomplete physical results should never produce a successful logical job."""
    plan = plan_jobs(make_payload())
    with pytest.raises(ValueError, match="shot"):
        Quel3ResultAssembler(plan.jobs).add(
            plan.executions[0].result_mapping,
            Quel3BackendExecutionResult(
                {}, {"R0": [np.zeros((4, 4))]}, {"sampling_period_ns": 0.8}
            ),
        )


@pytest.mark.parametrize("failure", ["duplicate", "metadata", "incomplete"])
def test_result_assembly_rejects_invalid_chunk_sequences(failure: str) -> None:
    """Restoration should reject repeated shots, changing metadata, and incomplete coverage."""
    prepared = plan_jobs(make_payload(), max_capture_samples=8)
    assembler = Quel3ResultAssembler(prepared.jobs)
    result = Quel3BackendExecutionResult(
        {}, {"R0": [np.zeros((2, 4))]}, {"sampling_period_ns": 0.8}
    )
    assembler.add(prepared.executions[0].result_mapping, result)

    if failure == "duplicate":
        with pytest.raises(ValueError, match="duplicate"):
            assembler.add(prepared.executions[0].result_mapping, result)
    elif failure == "metadata":
        changed = replace(result, config={"sampling_period_ns": 0.4})
        with pytest.raises(ValueError, match="metadata"):
            assembler.add(prepared.executions[1].result_mapping, changed)
    else:
        with pytest.raises(ValueError, match="Incomplete"):
            assembler.finish()


def test_split_results_preserve_data_when_driver_buffers_are_reused() -> None:
    """Collected shot chunks should preserve their values after driver buffer reuse."""
    plan = plan_jobs(make_payload(), max_capture_samples=8)
    assembler = Quel3ResultAssembler(plan.jobs)
    for index, run in enumerate(plan.executions):
        buffer = np.full((run.payload.n_iterations, 4), index + 1, dtype=np.complex128)
        assembler.add(
            run.result_mapping,
            Quel3BackendExecutionResult(
                {}, {"R0": [buffer]}, {"sampling_period_ns": 0.8}
            ),
        )
        buffer.fill(-1)
    np.testing.assert_array_equal(
        assembler.finish()[0].data["R0"][0][:, 0], [1, 1, 2, 2, 3]
    )


def test_time_limit_handles_decimal_boundary() -> None:
    """Floating-point duration arithmetic should not add a spurious shot chunk."""
    plan = plan_jobs(make_payload(samples=1, shots=6), max_execution_duration_ns=4.8)
    assert [run.payload.n_iterations for run in plan.executions] == [6]


def test_different_instrument_frequencies_split_multi_instrument_jobs() -> None:
    """A frequency conflict on any shared instrument should prevent packing."""
    first = make_payload()
    second = make_payload(alias="R1", frequency=5e9)
    combined = replace(
        first, fixed_timelines={**first.fixed_timelines, **second.fixed_timelines}
    )
    changed = replace(
        combined,
        fixed_timelines={
            **combined.fixed_timelines,
            "R1": replace(second.fixed_timelines["R1"], frequency_hz=5e9 + 1),
        },
    )
    cache = make_cache({"R0": "unit:trx_p00p01", "R1": "unit:trx_p02p03"})
    assert [
        tuple(fragment.job_index for fragment in run.result_mapping.fragments)
        for run in plan_jobs(combined, changed, cache=cache).executions
    ] == [(0,), (1,)]


def test_waveform_limit_is_per_instrument_and_ignores_unused_shapes() -> None:
    """Independent instrument libraries should each receive the full waveform budget."""
    first, second = make_payload(), make_payload(alias="R1")
    payload = replace(
        first,
        waveform_library={
            **first.waveform_library,
            "unused": Quel3Waveform(np.ones(100, dtype=np.complex128), 0.8),
        },
        fixed_timelines={**first.fixed_timelines, **second.fixed_timelines},
    )
    cache = make_cache({"R0": "unit:trx_p00p01", "R1": "unit:trx_p02p03"})
    run = plan_jobs(payload, cache=cache, max_waveform_samples=4).executions[0]
    assert run.resources.waveform_samples == {"R0": 4, "R1": 4}
    assert len(run.payload.waveform_library) == 1


def test_repeated_waveform_events_consume_one_library_entry() -> None:
    """Repeated references should not multiply waveform library storage."""
    payload = make_payload()
    timeline = payload.fixed_timelines["R0"]
    payload = replace(
        payload,
        fixed_timelines={
            "R0": replace(
                timeline,
                events=(
                    *timeline.events,
                    replace(timeline.events[0], start_offset_ns=5),
                ),
            )
        },
    )
    assert plan_jobs(payload, max_waveform_samples=4).executions[
        0
    ].resources.waveform_samples == {"R0": 4}


def test_capture_length_uses_ceiling_before_shot_multiplication() -> None:
    """Off-grid capture lengths should use the same sample ceiling as execution."""
    payload = make_payload()
    timeline = payload.fixed_timelines["R0"]
    payload = replace(
        payload,
        fixed_timelines={
            "R0": replace(
                timeline, capture_windows=(Quel3CaptureWindow("read", 0.1, 0.81),)
            )
        },
    )
    run = plan_jobs(payload).executions[0]
    assert run.resources.capture_samples == {"unit:rx_p00": 10}
    assert run.payload.fixed_timelines["R0"].capture_windows[
        0
    ].start_offset_ns == pytest.approx(0.8)


def test_empty_batch_and_generation_only_jobs_are_supported() -> None:
    """Empty batches and timelines without capture should remain executable."""
    assert plan_jobs().executions == ()
    payload = make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "R0": replace(payload.fixed_timelines["R0"], capture_windows=())
        },
    )
    assert plan_jobs(payload).executions[0].resources.capture_samples == {}


def test_jobs_with_different_capture_periods_remain_separate() -> None:
    """Different RAW sampling periods should not be merged into one result schema."""
    cache = make_cache(
        {"R0": "unit:trx_p00p01", "R1": "unit:trx_p02p03"}, {"R1": 400_000}
    )
    assert (
        len(plan_jobs(make_payload(), make_payload(alias="R1"), cache=cache).executions)
        == 2
    )


def test_job_with_mixed_capture_periods_is_rejected_before_execution() -> None:
    """A single job requiring incompatible result sampling periods should fail early."""
    first, second = make_payload(), make_payload(alias="R1")
    payload = replace(
        first, fixed_timelines={**first.fixed_timelines, **second.fixed_timelines}
    )
    cache = make_cache(
        {"R0": "unit:trx_p00p01", "R1": "unit:trx_p02p03"}, {"R1": 400_000}
    )
    with pytest.raises(ValueError, match="Capture aliases must agree"):
        plan_jobs(payload, cache=cache)


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


@pytest.mark.parametrize("config", [None, {}, {"quelware_endpoint": "worker-host"}])
def test_controller_config_without_execution_options_uses_defaults(config) -> None:
    """Omitted execution settings should preserve controller and runtime defaults."""
    controller = Quel3BackendController.from_config_mapping(config)
    assert controller.execution_options == Quel3ExecutionOptions()
    assert controller.quelware_endpoint == ("worker-host" if config else "localhost")


def test_controller_planning_uses_configured_limits_and_per_call_options(
    monkeypatch,
) -> None:
    """Controller plans should honor backend defaults and complete call overrides."""
    controller = Quel3BackendController.from_config_mapping(
        {"execution": {"max_capture_samples": 8}}
    )
    cached = list(make_cache().snapshot().values())
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: cached,
    )
    controller.refresh_instrument_cache()

    def fail_load():
        pytest.fail("Planning must not load the execution runtime.")

    monkeypatch.setattr(controller.execution_manager, "_load_quelware_api", fail_load)
    requests = [BackendExecutionRequest(payload=make_payload())]
    assert len(controller.plan_execution(requests).executions) == 3
    assert (
        len(
            controller.plan_execution(
                requests, execution_options=Quel3ExecutionOptions()
            ).executions
        )
        == 1
    )


@pytest.mark.parametrize("override", [None, Quel3ExecutionOptions()])
def test_controller_and_injected_manager_share_execution_defaults(
    monkeypatch, override
) -> None:
    """Controller and manager should use one default configuration and call overrides."""
    manager = Quel3ExecutionManager(
        sampling_period_ns=0.4,
        capture_decimation_factor=1,
        execution_options=Quel3ExecutionOptions(max_capture_samples=8),
    )
    controller = Quel3BackendController(
        execution_manager=manager, execution_options=override
    )
    cached = list(make_cache().snapshot().values())
    monkeypatch.setattr(
        controller.resource_reader, "read_instrument_infos", lambda **kwargs: cached
    )
    controller.refresh_instrument_cache()
    requests = [BackendExecutionRequest(payload=make_payload())]
    expected_runs = 3 if override is None else 1
    assert len(controller.plan_execution(requests).executions) == expected_runs
    assert (
        len(
            manager.plan_execution(
                requests=requests, instrument_cache=make_cache()
            ).executions
        )
        == expected_runs
    )
    assert (
        len(
            controller.plan_execution(
                requests, execution_options=Quel3ExecutionOptions()
            ).executions
        )
        == 1
    )
    assert len(controller.plan_execution(requests).executions) == expected_runs
    controller.execution_options = Quel3ExecutionOptions(max_capture_samples=12)
    assert len(controller.plan_execution(requests).executions) == 2
    assert (
        len(
            manager.plan_execution(
                requests=requests, instrument_cache=make_cache()
            ).executions
        )
        == 2
    )


@pytest.mark.parametrize("delays", [None, {}, {"unit": {"trx_p00p01": 20}}])
def test_jobs_with_distinct_cable_delays_keep_separate_executions(delays) -> None:
    """Jobs with different cable-delay settings should retain separate directives."""
    first = make_payload()
    second = replace(first, cable_delay_ns=delays)
    third = replace(first, cable_delay_ns={"unit": {"trx_p00p01": 40}})
    plan = plan_jobs(second, third)
    assert len(plan.executions) == 2
    assert [run.payload.cable_delay_ns for run in plan.executions] == [
        delays,
        third.cable_delay_ns,
    ]


def test_duration_limit_includes_final_shot_interval() -> None:
    """The appended shot interval should count toward each execution duration limit."""
    payload = replace(make_payload(), shot_interval_ns=3.2)
    plan = plan_jobs(payload, max_execution_duration_ns=12.8)
    assert [run.payload.n_iterations for run in plan.executions] == [2, 2, 1]
    assert [run.resources.duration_ns for run in plan.executions] == pytest.approx(
        [12.8, 12.8, 6.4], rel=0, abs=1e-9
    )
