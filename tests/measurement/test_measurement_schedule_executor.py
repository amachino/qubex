"""Tests for measurement schedule executor orchestration."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from qxpulse import PulseSchedule

from qubex.backend import (
    BackendExecutionRequest,
    BackendExecutor,
)
from qubex.backend.quel1 import Quel1BackendController, Quel1BackendRawResult
from qubex.measurement.adapters import MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_result_factory import MeasurementResultFactory
from qubex.measurement.measurement_schedule_executor import MeasurementScheduleExecutor
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


def test_execute_validates_builds_executes_and_creates_result() -> None:
    """Given executor inputs, when execute is called, then it validates, runs backend, and builds result."""
    called: dict[str, object] = {}
    request = BackendExecutionRequest(payload=object())
    backend_result = Quel1BackendRawResult(status={}, data={}, config={})
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())

    class _Adapter:
        def validate_schedule(
            self,
            schedule: MeasurementSchedule,
        ) -> None:
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

    class _Executor:
        def execute(self, *, request: BackendExecutionRequest) -> Quel1BackendRawResult:
            called["execute_request"] = request
            return backend_result

    class _ResultFactory:
        def create(self, **kwargs: object):  # type: ignore[no-untyped-def]
            called["result_kwargs"] = kwargs
            return expected

    backend_controller = type("_BC", (), {"box_config": {"shots": 2}})()
    executor = MeasurementScheduleExecutor(
        backend_executor=cast(BackendExecutor, _Executor()),
        measurement_backend_adapter=cast(MeasurementBackendAdapter, _Adapter()),
        measurement_result_factory=cast(MeasurementResultFactory, _ResultFactory()),
        backend_controller=cast(Quel1BackendController, backend_controller),
    )

    schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = _make_config()

    result = executor.execute(schedule=schedule, config=config)

    assert called["validated"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["execute_request"] is request
    result_kwargs = called["result_kwargs"]
    assert isinstance(result_kwargs, dict)
    typed_kwargs = cast(dict[str, Any], result_kwargs)
    assert typed_kwargs["backend_result"] is backend_result
    assert typed_kwargs["measurement_config"] is config
    assert typed_kwargs["device_config"] == {"shots": 2}
    assert typed_kwargs["sampling_period_ns"] == 2.0
    assert typed_kwargs["avg_sample_stride"] == 4
    assert result is expected


def test_create_default_passes_none_to_delegate_backend_defaults(monkeypatch) -> None:
    """Given default factory call, when creating executor, then backend defaults are delegated."""
    called: dict[str, object] = {}

    class _BackendExecutor:
        def __init__(
            self,
            *,
            backend_controller: object,
            execution_mode: str | None,
            clock_health_checks: bool | None,
        ) -> None:
            called["backend_controller"] = backend_controller
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1BackendExecutor",
        _BackendExecutor,
    )

    backend_controller = object()
    experiment_system = object()

    executor = MeasurementScheduleExecutor.create_default(
        backend_controller=cast(Quel1BackendController, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(executor, MeasurementScheduleExecutor)
    assert called["backend_controller"] is backend_controller
    assert called["execution_mode"] is None
    assert called["clock_health_checks"] is None


def test_create_default_passes_backend_constraint_profile_to_adapter(
    monkeypatch,
) -> None:
    """Given backend dt, when creating default executor, then adapter receives strict profile with that period."""
    called: dict[str, object] = {}

    class _BackendExecutor:
        def __init__(self, **kwargs: object) -> None:
            called["backend_executor_kwargs"] = kwargs

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
        "qubex.measurement.measurement_schedule_executor.Quel1BackendExecutor",
        _BackendExecutor,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1MeasurementBackendAdapter",
        _Adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.MeasurementResultFactory",
        _ResultFactory,
    )

    backend_controller = type("_BC", (), {"DEFAULT_SAMPLING_PERIOD": 4.0})()
    experiment_system = object()

    executor = MeasurementScheduleExecutor.create_default(
        backend_controller=cast(Quel1BackendController, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(executor, MeasurementScheduleExecutor)
    assert called["adapter_backend_controller"] is backend_controller
    assert called["adapter_experiment_system"] is experiment_system
    profile = called["adapter_constraint_profile"]
    assert isinstance(profile, MeasurementConstraintProfile)
    assert profile.sampling_period_ns == 4.0
    assert profile.enforce_block_alignment is True


def test_create_default_uses_relaxed_constraint_mode(monkeypatch) -> None:
    """Given relaxed mode hint, when creating default executor, then adapter receives relaxed profile."""
    called: dict[str, object] = {}

    class _BackendExecutor:
        def __init__(self, **kwargs: object) -> None:
            called["backend_executor_kwargs"] = kwargs

    class _Adapter:
        def __init__(
            self, *, constraint_profile: MeasurementConstraintProfile, **_: object
        ) -> None:
            called["adapter_constraint_profile"] = constraint_profile

    class _ResultFactory:
        def __init__(self, *, experiment_system: object) -> None:
            called["result_factory_experiment_system"] = experiment_system

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1BackendExecutor",
        _BackendExecutor,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1MeasurementBackendAdapter",
        _Adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.MeasurementResultFactory",
        _ResultFactory,
    )

    backend_controller = type(
        "_BC",
        (),
        {
            "DEFAULT_SAMPLING_PERIOD": 0.4,
            "MEASUREMENT_CONSTRAINT_MODE": "relaxed",
        },
    )()
    experiment_system = object()

    executor = MeasurementScheduleExecutor.create_default(
        backend_controller=cast(Quel1BackendController, backend_controller),
        experiment_system=cast(Any, experiment_system),
    )

    assert isinstance(executor, MeasurementScheduleExecutor)
    profile = called["adapter_constraint_profile"]
    assert isinstance(profile, MeasurementConstraintProfile)
    assert profile.sampling_period_ns == 0.4
    assert profile.enforce_block_alignment is False
    assert profile.require_workaround_capture is False


def test_create_default_prefers_backend_custom_factories(monkeypatch) -> None:
    """Given backend custom factories, when creating default executor, then custom executor and adapter are used."""
    called: dict[str, object] = {}

    class _BackendExecutor:
        def execute(self, *, request: BackendExecutionRequest) -> Quel1BackendRawResult:
            return Quel1BackendRawResult(status={}, data={}, config={})

    class _Adapter:
        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            return None

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            return BackendExecutionRequest(payload=object())

    class _ResultFactory:
        def create(self, **kwargs: object) -> object:
            return MeasurementResultConverter.from_multiple(_make_multiple_result())

    def _unexpected_backend_executor(**kwargs: object) -> object:
        raise AssertionError("Quel1BackendExecutor fallback should not be used.")

    def _unexpected_adapter(**kwargs: object) -> object:
        raise AssertionError("Quel1MeasurementBackendAdapter fallback should not be used.")

    def _unexpected_result_factory(**kwargs: object) -> object:
        raise AssertionError("MeasurementResultFactory fallback should not be used.")

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1BackendExecutor",
        _unexpected_backend_executor,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.Quel1MeasurementBackendAdapter",
        _unexpected_adapter,
    )
    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_executor.MeasurementResultFactory",
        _unexpected_result_factory,
    )

    class _Controller:
        box_config: ClassVar[dict[str, int]] = {"shots": 1}
        DEFAULT_SAMPLING_PERIOD: ClassVar[float] = 0.4

        def create_measurement_backend_executor(
            self,
            *,
            execution_mode: str | None,
            clock_health_checks: bool | None,
        ) -> _BackendExecutor:
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return _BackendExecutor()

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

    backend_controller = _Controller()
    experiment_system = object()
    executor = MeasurementScheduleExecutor.create_default(
        backend_controller=cast(Quel1BackendController, backend_controller),
        experiment_system=cast(Any, experiment_system),
        execution_mode="parallel",
        clock_health_checks=True,
    )

    assert isinstance(executor, MeasurementScheduleExecutor)
    assert called["execution_mode"] == "parallel"
    assert called["clock_health_checks"] is True
    assert called["experiment_system"] is experiment_system
    assert called["result_factory_experiment_system"] is experiment_system
    profile = called["constraint_profile"]
    assert isinstance(profile, MeasurementConstraintProfile)
    assert profile.sampling_period_ns == 0.4
