"""Tests for MeasurementExecutionService batch timeline packing behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from qxpulse import (
    Blank,
    PulseChannel,
    PulseSchedule,
    Rect,
    VirtualZ,
    get_sampling_period,
    set_sampling_period,
)

from qubex.backend import BackendExecutionRequest
from qubex.measurement.models import (
    CaptureData,
    MeasurementConfig,
    MeasurementResult,
    MeasurementSchedule,
)
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)


def _make_config() -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )


def _make_schedule(
    *,
    label: str,
    capture_start: float,
    capture_target: str,
    pulse_duration: float = 4.0,
) -> MeasurementSchedule:
    with PulseSchedule([label]) as pulse_schedule:
        if pulse_duration > 0.0:
            pulse_schedule.add(label, Rect(duration=pulse_duration, amplitude=0.1))
    return MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=[capture_target],
                    start_time=capture_start,
                    duration=1.0,
                )
            ]
        ),
    )


@dataclass
class _BackendResult:
    status: dict[str, object]
    data: dict[str, list[Any]]
    config: dict[str, object]


class _FakeBackend:
    sampling_period_ns: ClassVar[float] = 0.4
    CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4

    async def execute_batch_async(
        self,
        *,
        requests: list[BackendExecutionRequest],
    ) -> list[_BackendResult]:
        del requests
        return []


class _FakeRunner:
    def __init__(
        self,
        *,
        _backend_controller: _FakeBackend,
        **_: Any,
    ) -> None:
        self._measurement_backend_adapter = object()
        self.prepare_calls: list[MeasurementSchedule] = []
        self.executed_requests: list[BackendExecutionRequest] = []
        self.build_chunks: list[tuple[str, ...]] = []

    async def _execute_request(
        self, *, request: BackendExecutionRequest
    ) -> _BackendResult:
        self.executed_requests.append(request)
        schedule = cast(MeasurementSchedule, request.payload)
        visible_capture_count = sum(
            0 if capture.is_workaround else len(capture.channels)
            for capture in schedule.capture_schedule.captures
        )
        return _BackendResult(
            status={},
            data={
                "Q00": [
                    np.array(
                        [[float(index + 1) + 0.0j], [float(index + 1) + 0.0j]],
                        dtype=np.complex128,
                    )
                    for index in range(visible_capture_count)
                ],
            },
            config={"sampling_period_ns": 0.4},
        )

    async def execute_async(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        request = self._prepare_execution(schedule=schedule, config=config)
        backend_result = await self._execute_request(request=request)
        return self._build_result(backend_result=backend_result, config=config)

    def _prepare_execution(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Any | None = None,
    ) -> BackendExecutionRequest:
        del config
        del quel1_options
        self.prepare_calls.append(schedule)
        return BackendExecutionRequest(payload=cast(object, schedule))

    def _build_result(
        self,
        *,
        backend_result: object,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        assert isinstance(backend_result, _BackendResult)
        backend_data = cast(dict[str, list[Any]], backend_result.data)
        self.build_chunks.append(tuple(backend_data.keys()))
        return MeasurementResult(
            data={
                target: [
                    CaptureData.from_primary_data(
                        target=target,
                        data=np.asarray(value),
                        config=config,
                        sampling_period=0.4,
                    )
                    for value in values
                ]
                for target, values in backend_data.items()
            },
            measurement_config=config,
            device_config={
                "chunk_aliases": tuple(backend_data.keys()),
            },
        )


def _make_context() -> tuple[SimpleNamespace, _FakeBackend]:
    backend = _FakeBackend()
    context = SimpleNamespace(
        config_loader=SimpleNamespace(measurement_config={}),
        experiment_system=SimpleNamespace(
            target_registry=SimpleNamespace(
                measurement_output_label=lambda target: target,
            ),
        ),
        system_manager=SimpleNamespace(),
    )
    return context, backend


def _make_service(
    monkeypatch,
    backend: _FakeBackend,
) -> tuple[MeasurementExecutionService, list[_FakeRunner]]:
    runners: list[_FakeRunner] = []

    def _runner_factory(
        *,
        backend_controller: _FakeBackend,
        **kwargs: Any,
    ) -> _FakeRunner:
        _ = backend_controller
        del kwargs
        runner = _FakeRunner(_backend_controller=backend)
        runners.append(runner)
        return runner

    monkeypatch.setattr(
        "qubex.measurement.services.measurement_execution_service.MeasurementScheduleRunner",
        _runner_factory,
    )
    context, _ = _make_context()
    service = MeasurementExecutionService(
        context=cast(Any, context),
        session_service=cast(Any, SimpleNamespace(backend_controller=backend)),
        classifiers={},
    )
    return service, runners


def test_run_sweep_measurement_with_schedule_packing_merges_and_splits(
    monkeypatch,
) -> None:
    """Given pack option, run_sweep_measurement should execute one merged timeline and split results."""
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q00", capture_start=2.0, capture_target="Q00"
    )
    schedules = [first_schedule, second_schedule]

    backend = _FakeBackend()
    service, runners = _make_service(monkeypatch=monkeypatch, backend=backend)

    def _schedule_builder(value: object) -> MeasurementSchedule:
        return schedules[int(cast(float, value))]

    config = _make_config().model_copy(update={"schedule_packing_enabled": True})

    result = asyncio.run(
        service.run_sweep_measurement(
            schedule=_schedule_builder,
            sweep_values=[0, 1],
            config=config,
        )
    )

    runner = runners[0]
    assert len(runner.prepare_calls) == 1
    assert len(runner.executed_requests) == 1

    merged_schedule = runner.prepare_calls[-1]
    expected_shift = first_schedule.pulse_schedule.duration + config.shot_interval

    merged_captures = merged_schedule.capture_schedule.channels
    assert merged_captures["Q00"][0].start_time == 1.0
    assert merged_captures["Q00"][1].start_time == pytest.approx(2.0 + expected_shift)

    assert runner.build_chunks == [("Q00",)]
    assert list(result.results[0].data) == ["Q00"]
    assert list(result.results[1].data) == ["Q00"]
    assert np.asarray(result.results[0].data["Q00"][0].data).tolist() == [
        [1.0 + 0.0j],
        [1.0 + 0.0j],
    ]
    assert np.asarray(result.results[1].data["Q00"][0].data).tolist() == [
        [2.0 + 0.0j],
        [2.0 + 0.0j],
    ]


def test_run_sweep_measurement_splits_packed_timelines_by_repeated_duration(
    monkeypatch,
) -> None:
    """Given a packed timeline limit, run_sweep_measurement should split packed chunks."""
    schedules = [
        _make_schedule(label="Q00", capture_start=float(index), capture_target="Q00")
        for index in range(4)
    ]

    backend = _FakeBackend()
    service, runners = _make_service(monkeypatch=monkeypatch, backend=backend)

    def _schedule_builder(value: object) -> MeasurementSchedule:
        return schedules[int(cast(float, value))]

    config = _make_config().model_copy(
        update={
            "shot_interval": 2.0,
            "schedule_packing_enabled": True,
            "max_repeated_timeline_duration_ns": 24,
        }
    )

    result = asyncio.run(
        service.run_sweep_measurement(
            schedule=_schedule_builder,
            sweep_values=[0, 1, 2, 3],
            config=config,
        )
    )

    runner = runners[0]
    assert len(runner.prepare_calls) == 2
    assert len(runner.executed_requests) == 2
    assert [
        len(schedule.capture_schedule.captures) for schedule in runner.prepare_calls
    ] == [
        2,
        2,
    ]
    assert len(result.results) == 4


def test_merge_measurement_schedules_rebuilds_capture_channels_after_source_cache() -> (
    None
):
    """Given cached channels, merge should expose captures appended from later schedules."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q00", capture_start=2.0, capture_target="Q00"
    )
    _ = first_schedule.capture_schedule.channels

    merged_schedule = service._merge_measurement_schedules(  # noqa: SLF001
        schedules=[first_schedule, second_schedule],
        shot_interval=100.0,
    )

    merged_captures = merged_schedule.capture_schedule.channels
    assert set(merged_captures) == {"Q00"}
    assert len(merged_captures["Q00"]) == 2
    assert merged_captures["Q00"][1].start_time == pytest.approx(106.0)


def test_should_pack_measurement_schedules_rejects_different_labels() -> None:
    """Given different labels, packing should fall back to batch execution."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q01", capture_start=2.0, capture_target="Q01"
    )
    config = _make_config().model_copy(update={"schedule_packing_enabled": True})

    assert (
        service._should_pack_measurement_schedules(  # noqa: SLF001
            runner=cast(Any, _FakeRunner(_backend_controller=_FakeBackend())),
            schedules=[first_schedule, second_schedule],
            config=config,
        )
        is False
    )


def test_should_pack_measurement_schedules_rejects_different_sampling_periods() -> None:
    """Given different channel sampling periods, packing should fall back to batch execution."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    original_sampling_period = get_sampling_period()
    try:
        set_sampling_period(2.0)
        first_schedule = _make_schedule(
            label="Q00", capture_start=1.0, capture_target="Q00"
        )
        set_sampling_period(0.8)
        second_schedule = _make_schedule(
            label="Q00", capture_start=2.0, capture_target="Q00"
        )
    finally:
        set_sampling_period(original_sampling_period)
    config = _make_config().model_copy(update={"schedule_packing_enabled": True})

    assert (
        service._should_pack_measurement_schedules(  # noqa: SLF001
            runner=cast(Any, _FakeRunner(_backend_controller=_FakeBackend())),
            schedules=[first_schedule, second_schedule],
            config=config,
        )
        is False
    )


def test_merge_measurement_schedules_appends_same_channels() -> None:
    """Given matching labels, merge should append pulses with shifted captures."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q00", capture_start=2.0, capture_target="Q00"
    )

    merged_schedule = service._merge_measurement_schedules(  # noqa: SLF001
        schedules=[first_schedule, second_schedule],
        shot_interval=100.0,
    )

    q00_waveforms = merged_schedule.pulse_schedule.get_sequence(
        "Q00",
        copy=False,
    ).get_flattened_waveforms(apply_frame_shifts=False)
    assert len(q00_waveforms) == 3
    assert isinstance(q00_waveforms[1], Blank)
    assert q00_waveforms[1].duration == pytest.approx(100.0)
    assert merged_schedule.capture_schedule.channels["Q00"][1].start_time == (
        pytest.approx(106.0)
    )
    assert merged_schedule.pulse_schedule.is_valid()


def test_merge_measurement_schedules_resets_virtual_z_between_schedules() -> None:
    """Packing should keep one schedule's virtual Z from rotating the next schedule."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    with PulseSchedule(["Q00"]) as first_pulse_schedule:
        first_pulse_schedule.add("Q00", Rect(duration=4.0, amplitude=0.1))
        first_pulse_schedule.add("Q00", VirtualZ(np.pi / 2))
    with PulseSchedule(["Q00"]) as second_pulse_schedule:
        second_pulse_schedule.add("Q00", Rect(duration=4.0, amplitude=0.2))
    schedules = [
        MeasurementSchedule(
            pulse_schedule=pulse_schedule,
            capture_schedule=CaptureSchedule(captures=[]),
        )
        for pulse_schedule in (first_pulse_schedule, second_pulse_schedule)
    ]

    merged_schedule = service._merge_measurement_schedules(  # noqa: SLF001
        schedules=schedules,
        shot_interval=100.0,
    )

    merged_waveforms = merged_schedule.pulse_schedule.get_sequence(
        "Q00",
        copy=False,
    ).get_flattened_waveforms(apply_frame_shifts=True)
    pulses = [
        waveform for waveform in merged_waveforms if not isinstance(waveform, Blank)
    ]
    np.testing.assert_allclose(pulses[0].values, 0.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(pulses[1].values, 0.2, rtol=0.0, atol=1e-12)


def test_merge_measurement_schedules_preserves_per_schedule_transforms() -> None:
    """Packing should preserve each schedule's scale, phase, and detuning."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q00", capture_start=2.0, capture_target="Q00"
    )
    first_schedule = first_schedule.model_copy(
        update={
            "pulse_schedule": first_schedule.pulse_schedule.scaled(0.1)
            .shifted(0.2)
            .detuned(0.003)
        }
    )
    second_schedule = second_schedule.model_copy(
        update={
            "pulse_schedule": second_schedule.pulse_schedule.scaled(0.3)
            .shifted(0.4)
            .detuned(0.005)
        }
    )

    merged_schedule = service._merge_measurement_schedules(  # noqa: SLF001
        schedules=[first_schedule, second_schedule],
        shot_interval=100.0,
    )

    waveforms = merged_schedule.pulse_schedule.get_sequence(
        "Q00",
        copy=False,
    ).get_flattened_waveforms(apply_frame_shifts=False)
    pulses = [waveform for waveform in waveforms if not isinstance(waveform, Blank)]
    assert all(waveform.duration > 0.0 for waveform in waveforms)
    assert [pulse.scale for pulse in pulses] == pytest.approx([0.1, 0.3])
    assert [pulse.phase for pulse in pulses] == pytest.approx([0.2, 0.4])
    assert [pulse.detuning for pulse in pulses] == pytest.approx([0.003, 0.005])


def test_merge_measurement_schedules_preserves_first_channel_metadata() -> None:
    """Packing into an empty schedule should preserve first-channel metadata."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    original_sampling_period = get_sampling_period()
    try:
        set_sampling_period(2.0)
        with PulseSchedule(
            [
                PulseChannel(
                    label="Q00",
                    frequency=5.25,
                    target="Q00",
                    frame="drive",
                )
            ]
        ) as pulse_schedule:
            pulse_schedule.add("Q00", Rect(duration=4.0, amplitude=0.1))
    finally:
        set_sampling_period(original_sampling_period)
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    merged_schedule = service._merge_measurement_schedules(  # noqa: SLF001
        schedules=[schedule],
        shot_interval=100.0,
    )

    merged_pulse_schedule = merged_schedule.pulse_schedule
    assert merged_pulse_schedule.get_frequencies() == {"Q00": 5.25}
    assert merged_pulse_schedule.get_targets() == {"Q00": "Q00"}
    assert merged_pulse_schedule.get_frames() == {"Q00": "drive"}
    assert merged_pulse_schedule.get_sequence("Q00", copy=False).sampling_period == 2.0


def test_split_merged_measurement_result_uses_capture_order_contract() -> None:
    """Given repeated target captures, split should consume captures from the front."""
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    config = _make_config()
    merged_result = MeasurementResult(
        data={
            "Q00": [
                CaptureData.from_primary_data(
                    target="Q00",
                    data=np.array([[1.0 + 0.0j], [1.0 + 0.0j]], dtype=np.complex128),
                    config=config,
                    sampling_period=0.4,
                ),
                CaptureData.from_primary_data(
                    target="Q00",
                    data=np.array([[2.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
                    config=config,
                    sampling_period=0.4,
                ),
            ]
        },
        measurement_config=config,
        device_config={"device": "fake"},
    )

    results = service._split_merged_measurement_result(  # noqa: SLF001
        merged_result=merged_result,
        split_plan=[{"Q00": 1}, {"Q00": 1}],
    )

    assert np.asarray(results[0].data["Q00"][0].data).tolist() == [
        [1.0 + 0.0j],
        [1.0 + 0.0j],
    ]
    assert np.asarray(results[1].data["Q00"][0].data).tolist() == [
        [2.0 + 0.0j],
        [2.0 + 0.0j],
    ]
    assert results[0].device_config == {"device": "fake"}


def test_build_measurement_result_split_plan_uses_registry_output_labels() -> None:
    """Given registry output labels, split plan should use canonical result keys."""

    class _TargetRegistry:
        @staticmethod
        def measurement_output_label(target_label: str) -> str:
            return "Q17" if target_label == "raw-readout-target" else target_label

    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    cast(Any, service)._context = SimpleNamespace(  # noqa: SLF001
        experiment_system=SimpleNamespace(target_registry=_TargetRegistry())
    )
    schedule = _make_schedule(
        label="raw-readout-target",
        capture_start=1.0,
        capture_target="raw-readout-target",
    )

    split_plan = service._build_measurement_result_split_plan(  # noqa: SLF001
        schedules=[schedule],
    )

    assert split_plan == [{"Q17": 1}]


def test_should_pack_measurement_schedules_rejects_conflicting_frequencies() -> None:
    """Given per-point frequency changes, packing should fall back to batch execution."""
    backend = _FakeBackend()
    runner = _FakeRunner(_backend_controller=backend)
    service = MeasurementExecutionService.__new__(MeasurementExecutionService)
    first_schedule = _make_schedule(
        label="Q00", capture_start=1.0, capture_target="Q00"
    )
    second_schedule = _make_schedule(
        label="Q00", capture_start=2.0, capture_target="Q00"
    )
    first_schedule.pulse_schedule.set_frequency("Q00", 5.0)
    second_schedule.pulse_schedule.set_frequency("Q00", 5.1)
    config = _make_config().model_copy(update={"schedule_packing_enabled": True})

    assert (
        service._should_pack_measurement_schedules(  # noqa: SLF001
            runner=cast(Any, runner),
            schedules=[first_schedule, second_schedule],
            config=config,
        )
        is False
    )
