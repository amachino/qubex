"""Tests for measurement schedule runner orchestration."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, cast

import numpy as np
from qxpulse import PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1BackendRawResult
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


def _make_config() -> MeasurementConfig:
    return MeasurementConfig(
        mode="avg",
        shots=2,
        interval=100.0,
        frequencies={},
        enable_dsp_demodulation=True,
        enable_dsp_sum=False,
        enable_dsp_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
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
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

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

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0

        def execute(self, *, request: BackendExecutionRequest) -> Quel1BackendRawResult:
            called["execute_request"] = request
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            called["result_kwargs"] = kwargs
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
    )

    config = _make_config()
    schedule = _make_schedule()
    result = runner.execute(schedule=schedule, config=config)

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    result_kwargs = cast(dict[str, object], called["result_kwargs"])
    assert result_kwargs["backend_result"] is backend_result
    assert result_kwargs["measurement_config"] is config
    assert result_kwargs["device_config"] == {"shots": 2}
    assert result_kwargs["sampling_period_ns"] == 2.0
    assert result_kwargs["avg_sample_stride"] == 4
    assert result is expected


def test_execute_forwards_execution_options_to_backend_controller() -> None:
    """Given execution options, when execute is called, then backend controller receives options."""
    called: dict[str, object] = {}
    base_request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

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

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0

        def execute(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendRawResult:
            called["request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            _ = kwargs
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
        execution_mode="serial",
        clock_health_checks=True,
    )
    _ = runner.execute(schedule=_make_schedule(), config=_make_config())

    assert called["request"] is base_request
    assert called["execution_mode"] == "serial"
    assert called["clock_health_checks"] is True


def test_execute_async_validates_builds_calls_backend_and_creates_result() -> None:
    """Given runner inputs, when execute_async is called, then it validates, runs async backend controller, and builds result."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

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

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0

        async def execute_async(
            self, *, request: BackendExecutionRequest
        ) -> Quel1BackendRawResult:
            called["execute_request"] = request
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            called["result_kwargs"] = kwargs
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
    )

    config = _make_config()
    schedule = _make_schedule()
    result = asyncio.run(runner.execute_async(schedule=schedule, config=config))

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    result_kwargs = cast(dict[str, object], called["result_kwargs"])
    assert result_kwargs["backend_result"] is backend_result
    assert result_kwargs["measurement_config"] is config
    assert result_kwargs["device_config"] == {"shots": 2}
    assert result_kwargs["sampling_period_ns"] == 2.0
    assert result_kwargs["avg_sample_stride"] == 4
    assert result is expected


def test_execute_async_forwards_execution_options_to_backend_controller() -> None:
    """Given execution options, when execute_async is called, then backend controller receives options."""
    called: dict[str, object] = {}
    base_request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

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

    class _BackendController:
        box_config: ClassVar[dict[str, int]] = {"shots": 2}
        sampling_period: ClassVar[float] = 2.0

        async def execute_async(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendRawResult:
            called["request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            _ = kwargs
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
        execution_mode="serial",
        clock_health_checks=True,
    )
    _ = asyncio.run(
        runner.execute_async(schedule=_make_schedule(), config=_make_config())
    )

    assert called["request"] is base_request
    assert called["execution_mode"] == "serial"
    assert called["clock_health_checks"] is True


def test_execute_falls_back_to_empty_device_config_without_box_config() -> None:
    """Given backend without box config capability, when execute runs, then result factory receives an empty config."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

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

    class _BackendController:
        sampling_period: ClassVar[float] = 2.0

        def execute(self, *, request: BackendExecutionRequest) -> Quel1BackendRawResult:
            _ = request
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            called["result_kwargs"] = kwargs
            return expected

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
    )

    result = runner.execute(schedule=_make_schedule(), config=_make_config())

    result_kwargs = cast(dict[str, object], called["result_kwargs"])
    assert result_kwargs["device_config"] == {}
    assert result is expected


def test_execute_returns_backend_measurement_result_directly() -> None:
    """Given backend returns canonical result, when executing, then result factory is not called."""
    expected = MeasurementResult(
        mode="avg",
        data={"Q00": [np.array([1.0 + 0.0j])]},
        device_config={"kind": "quel3"},
        measurement_config={"mode": "avg"},
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

        def execute(self, *, request: BackendExecutionRequest) -> MeasurementResult:
            _ = request
            return expected

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            raise AssertionError("result factory should not be called")

    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=cast(Any, _Adapter()),
        measurement_result_factory=cast(Any, _ResultFactory()),
        backend_controller=cast(Any, _BackendController()),
    )
    result = runner.execute(schedule=_make_schedule(), config=_make_config())

    assert result is expected


def test_create_default_passes_backend_constraint_profile_to_adapter(
    monkeypatch,
) -> None:
    """Given backend dt, when creating default runner, then adapter receives strict profile with that period."""
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

    class _ResultFactory:
        def __init__(self, *, experiment_system: object) -> None:
            called["result_factory_experiment_system"] = experiment_system

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _Adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.MeasurementResultFactory",
        _ResultFactory,
    )

    backend_controller = type("_BC", (), {"sampling_period": 4.0})()
    experiment_system = object()

    runner = MeasurementScheduleRunner.create_default(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    assert called["adapter_backend_controller"] is backend_controller
    assert called["adapter_experiment_system"] is experiment_system
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 4.0
    assert profile.enforce_block_alignment is True


def test_create_default_uses_quel3_constraint_mode(monkeypatch) -> None:
    """Given quel3 mode hint, when creating default runner, then adapter receives quel3 profile."""
    called: dict[str, object] = {}

    class _Adapter:
        def __init__(
            self, *, constraint_profile: MeasurementConstraintProfile, **_: object
        ) -> None:
            called["adapter_constraint_profile"] = constraint_profile

    class _ResultFactory:
        def __init__(self, *, experiment_system: object) -> None:
            called["result_factory_experiment_system"] = experiment_system

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _Adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.MeasurementResultFactory",
        _ResultFactory,
    )

    backend_controller = type(
        "_BC",
        (),
        {
            "sampling_period": 0.4,
            "MEASUREMENT_CONSTRAINT_MODE": "quel3",
        },
    )()
    experiment_system = object()

    runner = MeasurementScheduleRunner.create_default(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is False
    assert profile.require_workaround_capture is False


def test_create_default_prefers_backend_custom_factories(monkeypatch) -> None:
    """Given backend custom factories, when creating default runner, then custom adapter and result factory are used."""
    called: dict[str, object] = {}

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            _ = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            called["request_schedule"] = schedule
            called["request_config"] = config
            return BackendExecutionRequest(payload=object())

    class _ResultFactory:
        def create(self, **kwargs: object) -> MeasurementResult:
            called["result_kwargs"] = kwargs
            return MeasurementResult(
                mode="avg",
                data={"Q00": [np.array([1.0 + 0.0j])]},
                device_config={"kind": "quel3"},
                measurement_config={"mode": "avg"},
            )

    def _unexpected_adapter(**kwargs: object) -> object:
        raise AssertionError(
            "Quel1MeasurementBackendAdapter fallback should not be used."
        )

    def _unexpected_result_factory(**kwargs: object) -> object:
        raise AssertionError("MeasurementResultFactory fallback should not be used.")

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _unexpected_adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.MeasurementResultFactory",
        _unexpected_result_factory,
    )

    class _Controller:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}
        sampling_period: ClassVar[float] = 0.4
        MEASUREMENT_CONSTRAINT_MODE: ClassVar[str] = "quel3"

        def create_measurement_backend_adapter(
            self,
            *,
            experiment_system: object,
            constraint_profile: MeasurementConstraintProfile,
        ) -> _Adapter:
            called["experiment_system"] = experiment_system
            called["constraint_profile"] = constraint_profile
            return _Adapter()

        def create_measurement_result_factory(
            self,
            *,
            experiment_system: object,
        ) -> _ResultFactory:
            called["result_factory_experiment_system"] = experiment_system
            return _ResultFactory()

        def execute(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendRawResult:
            called["execute_request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return Quel1BackendRawResult(status={}, data={}, config={})

    backend_controller = _Controller()
    experiment_system = object()
    runner = MeasurementScheduleRunner.create_default(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
        execution_mode="parallel",
        clock_health_checks=True,
    )
    result = runner.execute(schedule=_make_schedule(), config=_make_config())

    assert isinstance(runner, MeasurementScheduleRunner)
    assert called["experiment_system"] is experiment_system
    assert called["result_factory_experiment_system"] is experiment_system
    assert isinstance(called["constraint_profile"], MeasurementConstraintProfile)
    assert isinstance(called["execute_request"], BackendExecutionRequest)
    assert called["execution_mode"] == "parallel"
    assert called["clock_health_checks"] is True
    assert result.device_config == {"kind": "quel3"}


def test_create_default_uses_quel3_adapter_when_backend_kind_is_quel3(
    monkeypatch,
) -> None:
    """Given quel3 backend kind hint, when creating default runner, then Quel3 adapter is selected."""
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

    class _ResultFactory:
        def __init__(self, *, experiment_system: object) -> None:
            called["result_factory_experiment_system"] = experiment_system

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
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.MeasurementResultFactory",
        _ResultFactory,
    )

    backend_controller = type(
        "_BC",
        (),
        {
            "sampling_period": 0.4,
            "MEASUREMENT_CONSTRAINT_MODE": "quel3",
            "MEASUREMENT_BACKEND_KIND": "quel3",
        },
    )()
    experiment_system = object()

    runner = MeasurementScheduleRunner.create_default(
        backend_controller=cast(Any, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(runner, MeasurementScheduleRunner)
    assert called["adapter_backend_controller"] is backend_controller
    assert called["adapter_experiment_system"] is experiment_system
    profile = cast(MeasurementConstraintProfile, called["adapter_constraint_profile"])
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is False
