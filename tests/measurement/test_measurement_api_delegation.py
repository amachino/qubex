"""Tests for legacy API delegation to `run()`."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from qubex.measurement.measurement_client import MeasurementClient
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
        save_waveforms: bool = False,
    ) -> MeasurementResult:
        called["run_schedule"] = schedule
        called["run_config"] = config
        called["run_save_waveforms"] = save_waveforms
        return MeasurementResult.from_multiple(multiple)

    def fake_to_multiple(
        self: MeasurementClient, result: MeasurementResult
    ) -> MultipleMeasureResult:
        return result.to_multiple_measure_result(config={"shots": 1})

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
        save_waveforms=True,
    )

    assert result.mode == multiple.mode
    assert np.array_equal(result.data["Q00"][0].raw, multiple.data["Q00"][0].raw)
    assert called["build_schedule"] is pulse_schedule
    assert called["run_schedule"] is built_schedule
    assert called["run_save_waveforms"] is True
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
    assert called["kwargs"]["save_waveforms"] is False
    assert result.data["Q00"] is multiple.data["Q00"][0]


def test_run_delegates_schedule_execution_to_executor() -> None:
    """Given run inputs, when run is called, then it validates and executes via DeviceExecutor."""
    measurement = object.__new__(MeasurementClient)
    measurement.__dict__["_system_manager"] = type("_SM", (), {"rawdata_dir": None})()
    measurement.__dict__["_device_executor"] = type("_Exec", (), {})()

    pulse_schedule = PulseSchedule(["RQ00"])
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = MeasurementConfig.from_execute_args(mode="avg", shots=2, interval=100.0)
    expected = MeasurementResult.from_multiple(_make_multiple_result())
    called: dict[str, Any] = {}

    def fake_validate(self: object, schedule_arg: MeasurementSchedule) -> None:
        called["validated"] = schedule_arg

    def fake_execute_schedule(self: object, **kwargs: object) -> dict[str, object]:
        called["execute_kwargs"] = kwargs
        return {"data": {}, "status": {}, "config": {}}

    def fake_create_result(self: MeasurementClient, **_: object) -> MeasurementResult:
        return expected

    measurement.__dict__["_device_executor"].validate_schedule = MethodType(
        fake_validate,
        measurement.__dict__["_device_executor"],
    )
    measurement.__dict__["_device_executor"].execute_schedule = MethodType(
        fake_execute_schedule,
        measurement.__dict__["_device_executor"],
    )
    measurement._create_measurement_result = MethodType(  # noqa: SLF001
        fake_create_result,
        measurement,
    )

    result = measurement.run(schedule=schedule, config=config)

    assert called["validated"] is schedule
    assert called["execute_kwargs"]["schedule"] is schedule
    assert called["execute_kwargs"]["interval"] == config.interval
    assert result is expected
