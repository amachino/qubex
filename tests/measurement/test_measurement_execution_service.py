"""Tests for backend-neutral measurement batch delegation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from qxpulse import (
    PulseSchedule,
    Rect,
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

    async def execute_batch_async(self, *, schedules, config):
        return [
            await self.execute_async(schedule=schedule, config=config)
            for schedule in schedules
        ]

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


@pytest.mark.parametrize("frequencies", [(5.0, 5.0), (5.0, 5.1)])
def test_sweep_delegates_individual_schedules_to_backend_batch(
    monkeypatch, frequencies
):
    """Measurement should delegate intact schedules regardless of frequency differences."""
    schedules = [
        _make_schedule(label="Q00", capture_start=float(index), capture_target="Q00")
        for index in range(2)
    ]
    for schedule, frequency in zip(schedules, frequencies, strict=True):
        schedule.pulse_schedule.set_frequency("Q00", frequency)
    service, runners = _make_service(monkeypatch, _FakeBackend())
    result = asyncio.run(
        service.run_sweep_measurement(
            schedule=lambda value: schedules[int(value)],
            sweep_values=[0, 1],
            config=_make_config(),
        )
    )
    assert runners[0].prepare_calls == schedules
    assert len(runners[0].executed_requests) == 2
    assert len(result.results) == 2
    assert [
        schedule.capture_schedule.captures[0].start_time for schedule in schedules
    ] == [0, 1]
    assert [
        schedule.pulse_schedule.get_frequency("Q00") for schedule in schedules
    ] == list(frequencies)
