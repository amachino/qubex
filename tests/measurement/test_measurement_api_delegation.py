"""Tests for legacy API delegation to `run()`."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from qubex.measurement.measurement import Measurement
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
    measurement = object.__new__(Measurement)
    pulse_schedule = PulseSchedule(["Q00"])
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    multiple = _make_multiple_result()

    called: dict[str, Any] = {}

    def fake_build(
        self: Measurement,
        *,
        schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_schedule"] = schedule
        called["build_kwargs"] = kwargs
        return built_schedule

    def fake_run(
        self: Measurement,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["run_schedule"] = schedule
        called["run_config"] = config
        return MeasurementResult(multiple=multiple)

    measurement._build_measurement_schedule = MethodType(  # noqa: SLF001
        fake_build, measurement
    )
    measurement.run = MethodType(fake_run, measurement)

    result = measurement.execute(schedule=pulse_schedule, add_last_measurement=True)

    assert result is multiple
    assert called["build_schedule"] is pulse_schedule
    assert called["run_schedule"] is built_schedule
    assert called["build_kwargs"]["add_last_measurement"] is True
    assert called["run_config"].mode == "avg"


def test_measure_delegates_to_execute_and_returns_first_capture() -> None:
    """Given measure inputs, when measure is called, then it delegates to execute and flattens first capture."""
    measurement = object.__new__(Measurement)
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_execute(self: Measurement, **kwargs: object) -> MultipleMeasureResult:
        called["kwargs"] = kwargs
        return multiple

    measurement.execute = MethodType(fake_execute, measurement)

    result = measurement.measure(waveforms={"Q00": np.array([0.0 + 0.0j])})

    assert called["kwargs"]["add_last_measurement"] is True
    assert result.data["Q00"] is multiple.data["Q00"][0]
