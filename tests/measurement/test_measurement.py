"""Tests for Measurement orchestration behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qubex.measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    Measurement,
)
from qubex.measurement.models import MeasureData, MeasureMode, MultipleMeasureResult
from qubex.pulse import Blank, PulseSchedule


class StubScheduleBuilder:
    """Record schedule-builder interactions during execute."""

    def __init__(
        self,
        *,
        readout_targets: list[str],
        capture_schedule: Any = None,
    ) -> None:
        self._readout_targets = readout_targets
        self._capture_schedule = capture_schedule
        self.build_calls: list[dict[str, Any]] = []

    def build_measurement_schedule(self, **kwargs: Any) -> Any:
        """Record schedule build calls and return a synthetic schedule bundle."""
        self.build_calls.append(kwargs)
        schedule = kwargs["schedule"]
        if not self._readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")
        adjuster = kwargs.get("schedule_adjuster")
        if adjuster is not None:
            adjuster(schedule)
        return SimpleNamespace(
            pulse_schedule=schedule,
            capture_schedule=self._capture_schedule,
        )


class StubDeviceExecutor:
    """Record executor interactions during execute."""

    def __init__(self, *, execute_result: Any) -> None:
        self._execute_result = execute_result
        self.adjust_calls: list[PulseSchedule] = []
        self.execute_calls: list[dict[str, Any]] = []

    def adjust_schedule_for_device(self, schedule: PulseSchedule) -> None:
        """Record schedule adjustment calls."""
        self.adjust_calls.append(schedule)

    def execute(self, **kwargs: Any) -> Any:
        """Record execution calls and return a configured result."""
        self.execute_calls.append(kwargs)
        return self._execute_result


class _MeasurementForTest(Measurement):
    """Expose injectable schedule builder for orchestration tests."""

    _test_schedule_builder: Any

    @property
    def schedule_builder(self) -> Any:
        """Return the injected schedule builder."""
        return self._test_schedule_builder


def _make_schedule() -> PulseSchedule:
    """Create a minimal valid schedule containing one readout pulse."""
    schedule = PulseSchedule(["RQ00"])
    schedule.add("RQ00", Blank(16))
    return schedule


def _make_measurement(
    builder: StubScheduleBuilder,
    executor: StubDeviceExecutor,
) -> Measurement:
    """Create a Measurement instance with stub dependencies."""
    measurement = object.__new__(_MeasurementForTest)
    object.__setattr__(measurement, "_test_schedule_builder", builder)
    object.__setattr__(measurement, "_device_executor", executor)
    object.__setattr__(measurement, "_classifiers", {})
    object.__setattr__(measurement, "_system_manager", SimpleNamespace())
    return measurement


def test_execute_delegates_schedule_and_execution_steps() -> None:
    """Given valid schedule, when execute, then delegate builder and executor steps."""
    capture_schedule = object()
    expected_result = object()
    builder = StubScheduleBuilder(
        readout_targets=["RQ00"],
        capture_schedule=capture_schedule,
    )
    executor = StubDeviceExecutor(execute_result=expected_result)
    measurement = _make_measurement(builder, executor)
    schedule = _make_schedule()

    result = measurement.execute(
        schedule=schedule,
        mode="single",
        shots=7,
        interval=42.0,
        add_last_measurement=True,
        add_pump_pulses=True,
    )

    assert result is expected_result
    assert len(builder.build_calls) == 1
    assert builder.build_calls[0]["add_last_measurement"] is True
    assert builder.build_calls[0]["add_pump_pulses"] is True
    assert executor.adjust_calls == [schedule]
    assert len(executor.execute_calls) == 1
    assert executor.execute_calls[0]["schedule"] is schedule
    assert executor.execute_calls[0]["capture_schedule"] is capture_schedule
    assert executor.execute_calls[0]["measure_mode"] is MeasureMode.SINGLE
    assert executor.execute_calls[0]["shots"] == 7
    assert executor.execute_calls[0]["interval"] == 42.0


def test_execute_uses_default_shots_and_interval() -> None:
    """Given omitted shots and interval, when execute, then use measurement defaults."""
    builder = StubScheduleBuilder(readout_targets=["RQ00"], capture_schedule=object())
    executor = StubDeviceExecutor(execute_result=object())
    measurement = _make_measurement(builder, executor)

    measurement.execute(schedule=_make_schedule())

    assert len(executor.execute_calls) == 1
    assert executor.execute_calls[0]["shots"] == DEFAULT_SHOTS
    assert executor.execute_calls[0]["interval"] == DEFAULT_INTERVAL
    assert executor.execute_calls[0]["measure_mode"] is MeasureMode.AVG


def test_execute_raises_when_readout_target_is_missing() -> None:
    """Given no readout targets, when execute, then raise ValueError before adjust."""
    builder = StubScheduleBuilder(readout_targets=[], capture_schedule=object())
    executor = StubDeviceExecutor(execute_result=object())
    measurement = _make_measurement(builder, executor)

    with pytest.raises(ValueError, match="No readout targets"):
        measurement.execute(schedule=_make_schedule())

    assert executor.adjust_calls == []
    assert executor.execute_calls == []


def test_measure_returns_first_capture_per_target() -> None:
    """Given multiple captures, when measure, then return only first capture data."""
    measurement = object.__new__(Measurement)
    first = MeasureData(target="Q00", mode=MeasureMode.SINGLE, raw=np.array([[1 + 0j]]))
    second = MeasureData(
        target="Q00", mode=MeasureMode.SINGLE, raw=np.array([[2 + 0j]])
    )
    third = MeasureData(target="Q01", mode=MeasureMode.SINGLE, raw=np.array([[3 + 0j]]))
    multi = MultipleMeasureResult(
        mode=MeasureMode.SINGLE,
        data={"Q00": [first, second], "Q01": [third]},
        config={"dummy": 1},
    )

    def _execute_stub(*args: Any, **kwargs: Any) -> MultipleMeasureResult:
        return multi

    measurement.execute = _execute_stub  # type: ignore[method-assign]

    result = measurement.measure(waveforms={"Q00": np.array([0 + 0j])})

    assert result.mode is MeasureMode.SINGLE
    assert result.data["Q00"] is first
    assert result.data["Q01"] is third
    assert result.config == {"dummy": 1}


def test_reload_uses_loaded_config_paths() -> None:
    """Given existing config paths, when reload, then call load and connect with them."""
    config_path = Path("dummy/config")
    params_path = Path("dummy/params")
    measurement = object.__new__(Measurement)
    object.__setattr__(
        measurement,
        "_system_manager",
        SimpleNamespace(
            config_loader=SimpleNamespace(
                config_path=config_path,
                params_path=params_path,
            )
        ),
    )
    calls: dict[str, Any] = {}

    def _load_stub(
        *, config_dir: Path, params_dir: Path, configuration_mode: str
    ) -> None:
        calls["load"] = {
            "config_dir": config_dir,
            "params_dir": params_dir,
            "configuration_mode": configuration_mode,
        }

    def _connect_stub() -> None:
        calls["connect"] = True

    measurement.load = _load_stub  # type: ignore[method-assign]
    measurement.connect = _connect_stub  # type: ignore[method-assign]

    measurement.reload(configuration_mode="ge-ef-cr")

    assert calls["load"]["config_dir"] == config_path
    assert calls["load"]["params_dir"] == params_path
    assert calls["load"]["configuration_mode"] == "ge-ef-cr"
    assert calls["connect"] is True
