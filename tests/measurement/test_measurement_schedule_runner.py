"""Tests for measurement schedule runner orchestration."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from qxpulse import PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1BackendExecutionResult
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models import (
    MeasurementConfig,
    MeasurementResult,
    MeasurementSchedule,
)
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MultipleMeasureResult,
)
from qubex.typing import MeasurementMode


def _make_config(
    *,
    mode: MeasurementMode = "avg",
    shots: int = 2,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval_ns=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=False,
        state_classification=False,
    )


def _make_schedule() -> MeasurementSchedule:
    return MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )


def _make_multiple_result() -> MultipleMeasureResult:
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )
    return MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={"Q00": [data]},
        config={"shots": 2},
    )


def test_execute_validates_builds_calls_backend_and_creates_result() -> None:
    """Given runner inputs, when execute is called, then it validates, runs backend controller, and builds result."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendExecutionResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(
        _make_multiple_result(),
        measurement_config=_make_config(),
    )

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            called["validated"] = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            called["request_schedule"] = schedule
            called["request_config"] = config
            return request

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
        ) -> MeasurementResult:
            called["result_kwargs"] = {
                "backend_result": backend_result,
                "measurement_config": measurement_config,
                "device_config": device_config,
                "sampling_period_ns": sampling_period_ns,
            }
            return expected

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4

        async def execute(
            self, *, request: BackendExecutionRequest
        ) -> Quel1BackendExecutionResult:
            called["execute_request"] = request
            return backend_result

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
    )

    config = _make_config()
    schedule = _make_schedule()
    result = asyncio.run(runner.execute(schedule=schedule, config=config))

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    result_kwargs = cast(dict[str, object], called["result_kwargs"])
    assert result_kwargs["backend_result"] is backend_result
    assert result_kwargs["measurement_config"] is config
    assert result_kwargs["device_config"] == {"shots": 2}
    assert result_kwargs["sampling_period_ns"] == 8.0
    assert result is expected


def test_execute_forwards_execution_options_to_backend_controller() -> None:
    """Given execution options, when execute is called, then backend controller receives options."""
    called: dict[str, object] = {}
    base_request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendExecutionResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(
        _make_multiple_result(),
        measurement_config=_make_config(),
    )

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            _ = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            _ = schedule
            _ = config
            return base_request

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
        ) -> MeasurementResult:
            _ = backend_result
            _ = measurement_config
            _ = device_config
            _ = sampling_period_ns
            return expected

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4

        async def execute(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendExecutionResult:
            called["request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return backend_result

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
        execution_mode="serial",
        clock_health_checks=True,
    )
    _ = asyncio.run(runner.execute(schedule=_make_schedule(), config=_make_config()))

    assert called["request"] is base_request
    assert called["execution_mode"] == "serial"
    assert called["clock_health_checks"] is True


def test_execute_falls_back_to_empty_device_config_without_box_config() -> None:
    """Given backend without box config capability, when execute runs, then adapter result builder receives an empty config."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendExecutionResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(
        _make_multiple_result(),
        measurement_config=_make_config(),
    )

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            _ = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            _ = schedule
            _ = config
            return request

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
        ) -> MeasurementResult:
            called["result_kwargs"] = {
                "backend_result": backend_result,
                "measurement_config": measurement_config,
                "device_config": device_config,
                "sampling_period_ns": sampling_period_ns,
            }
            return expected

    class _BackendController:
        sampling_period: ClassVar[float] = 2.0
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4

        async def execute(
            self, *, request: BackendExecutionRequest
        ) -> Quel1BackendExecutionResult:
            _ = request
            return backend_result

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
    )

    result = asyncio.run(
        runner.execute(schedule=_make_schedule(), config=_make_config())
    )

    result_kwargs = cast(dict[str, object], called["result_kwargs"])
    assert result_kwargs["device_config"] == {}
    assert result is expected


def test_execute_prefers_backend_capture_decimation_hint() -> None:
    """Given backend capture decimation, when execute runs, then adapter receives effective sampling metadata."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendExecutionResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(
        _make_multiple_result(),
        measurement_config=_make_config(),
    )

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            _ = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            _ = schedule
            _ = config
            return request

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
        ) -> MeasurementResult:
            _ = backend_result
            _ = measurement_config
            _ = device_config
            called["sampling_period_ns"] = sampling_period_ns
            return expected

    class _BackendController:
        sampling_period: ClassVar[float] = 2.0
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 8

        async def execute(
            self, *, request: BackendExecutionRequest
        ) -> Quel1BackendExecutionResult:
            _ = request
            return backend_result

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
    )

    result = asyncio.run(
        runner.execute(schedule=_make_schedule(), config=_make_config())
    )

    assert called["sampling_period_ns"] == 16.0
    assert result is expected


def test_execute_returns_backend_measurement_result_directly() -> None:
    """Given backend returns canonical result, when executing, then result factory is not called."""
    expected = MeasurementResult(
        data={"Q00": [np.array([1.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg"),
        device_config={"kind": "quel3"},
    )

    class _Adapter:
        sampling_period = 0.4
        constraint_profile = MeasurementConstraintProfile.quel3(0.4)

        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            _ = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            _ = schedule
            _ = config
            return BackendExecutionRequest(payload=object())

    class _BackendController:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}

        async def execute(
            self, *, request: BackendExecutionRequest
        ) -> MeasurementResult:
            _ = request
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
    )
    result = asyncio.run(
        runner.execute(schedule=_make_schedule(), config=_make_config())
    )

    assert result is expected


def test_execute_prefers_adapter_measurement_result_builder_when_available() -> None:
    """Given adapter result builder, when execute is called, then runner uses adapter conversion."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendExecutionResult(status={}, data={}, config={})
    expected = MeasurementResult(
        data={"Q00": [np.array([2.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg"),
        device_config={"kind": "adapter"},
    )

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            called["validated"] = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            called["request_schedule"] = schedule
            called["request_config"] = config
            return request

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
        ) -> MeasurementResult:
            called["builder_backend_result"] = backend_result
            called["builder_measurement_config"] = measurement_config
            called["builder_device_config"] = device_config
            called["builder_sampling_period_ns"] = sampling_period_ns
            return expected

    class _BackendController:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}
        sampling_period: ClassVar[float] = 0.4
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4

        async def execute(
            self, *, request: BackendExecutionRequest
        ) -> Quel1BackendExecutionResult:
            called["execute_request"] = request
            return backend_result

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        backend_controller=cast(Any, _BackendController()),
    )

    config = _make_config()
    schedule = _make_schedule()
    result = asyncio.run(runner.execute(schedule=schedule, config=config))

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    assert called["builder_backend_result"] is backend_result
    assert called["builder_measurement_config"] is config
    assert called["builder_device_config"] == {"kind": "quel3"}
    assert called["builder_sampling_period_ns"] == 1.6
    assert result is expected


def test_init_sets_default_adapter_and_constraint_profile_for_quel1(
    monkeypatch,
) -> None:
    """Given quel1 backend, when adapter is omitted, then init creates quel1 adapter with strict profile."""
    called: dict[str, object] = {}

    class _Adapter:
        def __init__(
            self,
            *,
            backend_controller: object,
            experiment_system: object,
            constraint_profile: MeasurementConstraintProfile,
        ) -> None:
            called["adapter_backend_controller"] = backend_controller
            called["adapter_experiment_system"] = experiment_system
            called["adapter_constraint_profile"] = constraint_profile

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _Adapter,
    )

    class _Quel1Controller:
        sampling_period: ClassVar[float] = 4.0

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1BackendController",
        _Quel1Controller,
    )

    backend_controller = _Quel1Controller()
    experiment_system = object()

    runner = MeasurementScheduleRunner(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    assert called["adapter_backend_controller"] is backend_controller
    assert called["adapter_experiment_system"] is experiment_system
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 4.0
    assert profile.enforce_block_alignment is True


def test_init_ignores_constraint_mode_hint_for_quel1(
    monkeypatch,
) -> None:
    """Given quel1 backend type, when adapter is omitted, then init still applies quel1 constraints."""
    called: dict[str, object] = {}

    class _Adapter:
        def __init__(
            self, *, constraint_profile: MeasurementConstraintProfile, **_: object
        ) -> None:
            called["adapter_constraint_profile"] = constraint_profile

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _Adapter,
    )

    class _Quel1Controller:
        sampling_period: ClassVar[float] = 0.4
        MEASUREMENT_CONSTRAINT_MODE: ClassVar[str] = "quel3"

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1BackendController",
        _Quel1Controller,
    )

    backend_controller = _Quel1Controller()
    experiment_system = object()

    runner = MeasurementScheduleRunner(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is True
    assert profile.require_workaround_capture is True


def test_init_raises_for_unsupported_backend_controller_without_adapter() -> None:
    """Given unsupported backend controller, when adapter is omitted, then init raises TypeError."""
    backend_controller = type("_UnsupportedController", (), {"sampling_period": 0.4})()

    with pytest.raises(TypeError, match="Unsupported backend controller"):
        _ = MeasurementScheduleRunner(
            backend_controller=cast(Any, backend_controller),
            experiment_system=cast(Any, object()),
        )


def test_init_requires_experiment_system_when_adapter_is_omitted() -> None:
    """Given omitted adapter without experiment system, when init runs, then it raises ValueError."""

    class _Quel1Controller:
        sampling_period: ClassVar[float] = 0.4

    with pytest.raises(ValueError, match="experiment_system"):
        _ = MeasurementScheduleRunner(
            backend_controller=cast(Any, _Quel1Controller()),
            measurement_backend_adapter=None,
        )


def test_init_uses_quel3_adapter_when_backend_controller_is_quel3(
    monkeypatch,
) -> None:
    """Given quel3 backend controller, when adapter is omitted, then init selects quel3 adapter."""
    called: dict[str, object] = {}

    class _Quel3Adapter:
        def __init__(
            self,
            *,
            backend_controller: object,
            experiment_system: object,
            constraint_profile: MeasurementConstraintProfile,
        ) -> None:
            called["adapter_backend_controller"] = backend_controller
            called["adapter_experiment_system"] = experiment_system
            called["adapter_constraint_profile"] = constraint_profile

    def _unexpected_quel1_adapter(**kwargs: object) -> object:
        raise AssertionError("Quel1 adapter fallback should not be used for quel3.")

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _unexpected_quel1_adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel3MeasurementBackendAdapter",
        _Quel3Adapter,
    )

    class _Quel3Controller:
        sampling_period: ClassVar[float] = 0.4

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel3BackendController",
        _Quel3Controller,
    )

    backend_controller = _Quel3Controller()
    experiment_system = object()

    runner = MeasurementScheduleRunner(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    assert called["adapter_backend_controller"] is backend_controller
    assert called["adapter_experiment_system"] is experiment_system
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is False
