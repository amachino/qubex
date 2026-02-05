"""Tests for legacy API delegation to schedule execution APIs."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from qubex.measurement.measurement_client import MeasurementClient
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.models import (
    DspConfig,
    FrequencyConfig,
    MeasurementConfig,
    MeasurementSchedule,
)
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.pulse import PulseSchedule


def _make_config() -> MeasurementConfig:
    return MeasurementConfig(
        mode="avg",
        shots=2,
        interval=100.0,
        frequency=FrequencyConfig(frequencies={}),
        dsp=DspConfig(
            enable_dsp_demodulation=True,
            enable_dsp_sum=False,
            enable_dsp_classification=False,
            line_param0=(1.0, 0.0, 0.0),
            line_param1=(0.0, 1.0, 0.0),
        ),
    )


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


def test_execute_delegates_to_schedule_executor_with_built_schedule() -> None:
    """Given execute inputs, when execute is called, then it builds schedule and delegates to schedule execution."""
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
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_schedule"] = pulse_schedule
        called["build_kwargs"] = kwargs
        return built_schedule

    def fake_execute_measurement_schedule(
        self: MeasurementClient,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["run_schedule"] = schedule
        called["run_config"] = config
        return MeasurementResultConverter.from_multiple(multiple)

    measurement.build_measurement_schedule = MethodType(fake_build, measurement)
    measurement.execute_measurement_schedule = MethodType(
        fake_execute_measurement_schedule, measurement
    )
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {},
        },
    )()
    measurement.__dict__["_backend_manager"] = type(
        "_BM",
        (),
        {
            "device_controller": type("_DC", (), {"box_config": {"shots": 1}})(),
            "experiment_system": experiment_system,
        },
    )()

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


def test_execute_measurement_schedule_delegates_to_executor() -> None:
    """Given schedule execution inputs, when method is called, then it delegates to executor."""
    measurement = object.__new__(MeasurementClient)

    pulse_schedule = PulseSchedule(["RQ00"])
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = _make_config()
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())
    called: dict[str, Any] = {}

    class _Executor:
        def execute(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> MeasurementResult:
            called["schedule"] = schedule
            called["config"] = config
            return expected

    measurement.__dict__["_measurement_schedule_executor"] = _Executor()

    result = measurement.execute_measurement_schedule(schedule=schedule, config=config)

    assert called["schedule"] is schedule
    assert called["config"] is config
    assert result is expected
