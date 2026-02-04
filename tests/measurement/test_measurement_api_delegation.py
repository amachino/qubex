"""Tests for legacy API delegation to `run()`."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from qubex.measurement.measurement_client import MeasurementClient
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.models import MeasurementConfig, MeasurementSchedule
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.pulse import PulseSchedule


def _make_multiple_result() -> MultipleMeasureResult:
    data0 = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )
    return MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={"Q00": [data0]},
        config={"shots": 1},
    )


def test_execute_delegates_to_run_with_built_schedule() -> None:
    """Given execute inputs, when execute is called, then it builds schedule and delegates to run."""
    measurement = object.__new__(MeasurementClient)
    measurement.__dict__["_classifiers"] = {}
    pulse_schedule = PulseSchedule(["Q00"])
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    multiple = _make_multiple_result()

    called: dict[str, Any] = {}

    def fake_build(
        self: MeasurementClient,
        *,
        schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_schedule"] = schedule
        called["build_kwargs"] = kwargs
        return built_schedule

    def fake_run(
        self: MeasurementClient,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["run_schedule"] = schedule
        called["run_config"] = config
        return MeasurementResultConverter.from_multiple(multiple)

    def fake_to_multiple(
        self: MeasurementClient, result: MeasurementResult
    ) -> MultipleMeasureResult:
        return MeasurementResultConverter.to_multiple_measure_result(
            result,
            config={"shots": 1},
        )

    measurement._build_measurement_schedule = MethodType(  # noqa: SLF001
        fake_build, measurement
    )
    measurement.run = MethodType(fake_run, measurement)
    measurement._to_multiple_measure_result = MethodType(  # noqa: SLF001
        fake_to_multiple, measurement
    )

    result = measurement.execute(
        schedule=pulse_schedule,
        add_last_measurement=True,
        save_result=False,
    )

    assert result.mode == multiple.mode
    assert np.array_equal(result.data["Q00"][0].raw, multiple.data["Q00"][0].raw)
    assert called["build_schedule"] is pulse_schedule
    assert called["run_schedule"] is built_schedule
    assert called["build_kwargs"]["add_last_measurement"] is True
    assert called["run_config"].mode == "avg"


def test_measure_delegates_to_execute_and_returns_first_capture() -> None:
    """Given measure inputs, when measure is called, then it delegates to execute and flattens first capture."""
    measurement = object.__new__(MeasurementClient)
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_execute(
        self: MeasurementClient, **kwargs: object
    ) -> MultipleMeasureResult:
        called["kwargs"] = kwargs
        return multiple

    measurement.execute = MethodType(fake_execute, measurement)

    result = measurement.measure(waveforms={"Q00": np.array([0.0 + 0.0j])})

    assert called["kwargs"]["add_last_measurement"] is True
    assert result.data["Q00"] is multiple.data["Q00"][0]


def test_run_delegates_schedule_execution_to_executor() -> None:
    """Given run inputs, when run is called, then it validates and executes via BackendExecutor."""
    measurement = object.__new__(MeasurementClient)
    measurement.__dict__["_system_manager"] = type("_SM", (), {"rawdata_dir": None})()
    measurement.__dict__["_backend_manager"] = type(
        "_DM",
        (),
        {"device_controller": type("_DC", (), {"box_config": {"shots": 2}})()},
    )()

    pulse_schedule = PulseSchedule(["RQ00"])
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = MeasurementConfig.create(mode="avg", shots=2, interval=100.0)
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())
    called: dict[str, Any] = {}

    request_obj = object()

    class _Adapter:
        def validate_schedule(self, schedule_arg: MeasurementSchedule) -> None:
            called["validated"] = schedule_arg

        def build_execution_request(self, **kwargs: object) -> object:
            called["request_kwargs"] = kwargs
            return request_obj

    class _Exec:
        def execute(self, **kwargs: object) -> dict[str, object]:
            called["execute_kwargs"] = kwargs
            return {"data": {}, "status": {}, "config": {}}

    class _ResultFactory:
        def create(self, **kwargs: object) -> MeasurementResult:
            called["result_kwargs"] = kwargs
            return expected

    measurement.__dict__["_measurement_backend_adapter"] = _Adapter()
    measurement.__dict__["_backend_executor"] = _Exec()
    measurement.__dict__["_measurement_result_factory"] = _ResultFactory()

    result = measurement.run(schedule=schedule, config=config)

    assert called["validated"] is schedule
    assert called["request_kwargs"]["schedule"] is schedule
    assert called["request_kwargs"]["config"] is config
    assert called["execute_kwargs"]["request"] is request_obj
    assert called["result_kwargs"]["measurement_config"] is config
    assert result is expected
