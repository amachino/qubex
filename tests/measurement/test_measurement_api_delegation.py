"""Tests for legacy API delegation to schedule execution APIs."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import MethodType
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from qxpulse import PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1BackendResult
from qubex.measurement.measurement import Measurement
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models import (
    MeasurementConfig,
    MeasurementSchedule,
    SweepMeasurementConfig,
)
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)
from qubex.typing import TargetMap


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


def _bind_runtime(
    measurement: Measurement,
    *,
    backend_controller: object,
    experiment_system: object,
    rawdata_dir: object = None,
) -> None:
    context = type(
        "_CTX",
        (),
        {
            "backend_controller": backend_controller,
            "experiment_system": experiment_system,
            "mux_dict": {},
            "system_manager": type("_SM", (), {"rawdata_dir": rawdata_dir})(),
        },
    )()
    session_service = type(
        "_SS",
        (),
        {
            "backend_controller": backend_controller,
        },
    )()
    measurement.__dict__["_context"] = context
    measurement.__dict__["_session_service"] = session_service
    measurement.execution_service.__dict__["_context"] = context
    measurement.execution_service.__dict__["_session_service"] = session_service


def test_execute_delegates_to_schedule_executor_with_built_schedule() -> None:
    """Given execute inputs, when execute is called, then it builds schedule and delegates to schedule execution."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    pulse_schedule = PulseSchedule(["Q00"])
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    multiple = _make_multiple_result()

    called: dict[str, Any] = {}

    def fake_build(
        self: MeasurementExecutionService,
        *,
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_schedule"] = pulse_schedule
        called["build_kwargs"] = kwargs
        return built_schedule

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["run_schedule"] = schedule
        called["run_config"] = config
        return MeasurementResultConverter.from_multiple(multiple)

    execution_service = measurement.execution_service
    execution_service.build_measurement_schedule = MethodType(
        fake_build, execution_service
    )
    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {},
        },
    )()
    backend_controller = type("_BC", (), {"box_config": {"shots": 1}})()
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
        rawdata_dir=None,
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
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_execute(
        self: MeasurementExecutionService, **kwargs: object
    ) -> MultipleMeasureResult:
        called["kwargs"] = kwargs
        return multiple

    measurement.execution_service.execute = MethodType(
        fake_execute,
        measurement.execution_service,
    )

    result = measurement.measure(waveforms={"Q00": np.array([0.0 + 0.0j])})

    assert called["kwargs"]["add_last_measurement"] is True
    assert result.data["Q00"] is multiple.data["Q00"][0]


def test_measure_async_entrypoint_is_removed() -> None:
    """Given measurement facade, measure_async entrypoint is not exposed."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    assert not hasattr(measurement, "measure_async")


def test_measure_initializes_optional_flags_with_measure_defaults() -> None:
    """Given None optional flags, when measure is called, then it applies measure defaults."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_execute(
        self: MeasurementExecutionService, **kwargs: object
    ) -> MultipleMeasureResult:
        called["kwargs"] = kwargs
        return multiple

    measurement.execution_service.execute = MethodType(
        fake_execute,
        measurement.execution_service,
    )

    measurement.measure(
        waveforms={"Q00": np.array([0.0 + 0.0j])},
        add_pump_pulses=None,
        enable_dsp_demodulation=None,
        enable_dsp_classification=None,
    )

    kwargs = called["kwargs"]
    assert kwargs["add_pump_pulses"] is False
    assert kwargs["enable_dsp_demodulation"] is True
    assert kwargs["enable_dsp_sum"] is True
    assert kwargs["enable_dsp_classification"] is False


def test_measure_noise_disables_dsp_sum_by_default() -> None:
    """Given noise measurement inputs, when measure_noise is called, then DSP summation is disabled by default."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, Any] = {}
    expected = object()

    def fake_measure(self: MeasurementExecutionService, **kwargs: object) -> Any:
        called["kwargs"] = kwargs
        return expected

    measurement.execution_service.measure = MethodType(
        fake_measure,
        measurement.execution_service,
    )

    result = measurement.measure_noise(["Q00"], duration=1024.0)

    assert result is expected
    kwargs = called["kwargs"]
    assert kwargs["enable_dsp_sum"] is False
    assert kwargs["readout_duration"] == 1024.0
    assert kwargs["readout_amplitudes"] == {"Q00": 0}


def test_execute_initializes_optional_flags_with_execute_defaults() -> None:
    """Given None optional flags, when execute is called, then it applies execute defaults."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    pulse_schedule = PulseSchedule(["Q00"])
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_build(
        self: MeasurementExecutionService,
        *,
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_kwargs"] = kwargs
        return built_schedule

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["config"] = config
        return MeasurementResultConverter.from_multiple(multiple)

    execution_service = measurement.execution_service
    execution_service.build_measurement_schedule = MethodType(
        fake_build, execution_service
    )
    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )
    experiment_system = type(
        "_ES",
        (),
        {
            "control_params": type("_CP", (), {"readout_amplitude": {}})(),
            "measurement_defaults": {},
        },
    )()
    backend_controller = type("_BC", (), {"box_config": {"shots": 1}})()
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
        rawdata_dir=None,
    )

    measurement.execute(
        schedule=pulse_schedule,
        add_pump_pulses=None,
        enable_dsp_demodulation=None,
        enable_dsp_classification=None,
        save_result=False,
    )

    assert called["build_kwargs"]["add_pump_pulses"] is False
    config = called["config"]
    assert config.enable_dsp_demodulation is True
    assert config.enable_dsp_sum is True
    assert config.enable_dsp_classification is False


def test_run_measurement_delegates_to_executor(
    monkeypatch,
) -> None:
    """Given schedule execution inputs, when method is called, then it delegates to executor."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    pulse_schedule = PulseSchedule(["RQ00"])
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = _make_config()
    expected = MeasurementResultConverter.from_multiple(_make_multiple_result())
    called: dict[str, Any] = {}

    class _Executor:
        async def execute(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> MeasurementResult:
            called["schedule"] = schedule
            called["config"] = config
            return expected

    experiment_system = type("_ES", (), {})()
    backend_controller = type("_BC", (), {})()
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    monkeypatch.setattr(
        MeasurementScheduleRunner,
        "create_default",
        classmethod(
            lambda cls,
            *,
            backend_controller,
            experiment_system,
            execution_mode="serial",
            clock_health_checks=False: _Executor()
        ),
    )
    result = asyncio.run(measurement.run_measurement(schedule=schedule, config=config))

    assert called["schedule"] is schedule
    assert called["config"] is config
    assert result is expected


def test_run_measurement_async_entrypoint_is_removed() -> None:
    """Given measurement facade, run_measurement_async entrypoint is not exposed."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    assert not hasattr(measurement, "run_measurement_async")


def test_run_measurement_selects_quel3_adapter_from_controller_type(
    monkeypatch,
) -> None:
    """Given quel3 backend controller type, when executing a schedule, then Quel3 adapter is selected."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    pulse_schedule = PulseSchedule(["RQ00"])
    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = _make_config()
    called: dict[str, object] = {}

    def _unexpected_adapter(**kwargs: object) -> object:
        raise AssertionError(
            "Quel1MeasurementBackendAdapter fallback should not be used."
        )

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel1MeasurementBackendAdapter",
        _unexpected_adapter,
    )

    class _Quel3Adapter:
        def __init__(
            self,
            *,
            experiment_system: object,
            constraint_profile: object,
            **_: object,
        ) -> None:
            called["experiment_system"] = experiment_system
            called["constraint_profile"] = constraint_profile

        def validate_schedule(self, schedule: MeasurementSchedule) -> None:
            called["validated_schedule"] = schedule

        def build_execution_request(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
        ) -> BackendExecutionRequest:
            called["request_schedule"] = schedule
            called["request_config"] = config
            return BackendExecutionRequest(payload=object())

        def build_measurement_result(
            self,
            *,
            backend_result: object,
            measurement_config: MeasurementConfig,
            device_config: dict[str, object],
            sampling_period_ns: float,
            avg_sample_stride: int,
        ) -> MeasurementResult:
            called["result_kwargs"] = {
                "backend_result": backend_result,
                "measurement_config": measurement_config,
                "device_config": device_config,
                "sampling_period_ns": sampling_period_ns,
                "avg_sample_stride": avg_sample_stride,
            }
            return MeasurementResult(
                mode="avg",
                data={"Q00": [np.array([1.0 + 0.0j])]},
                device_config={"kind": "quel3"},
                measurement_config={"mode": "avg"},
            )

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel3MeasurementBackendAdapter",
        _Quel3Adapter,
    )

    class _Quel3Controller:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}
        sampling_period: ClassVar[float] = 0.4

        async def execute(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendResult:
            called["request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return Quel1BackendResult(status={}, data={}, config={})

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel3BackendController",
        _Quel3Controller,
    )

    experiment_system = object()
    _bind_runtime(
        measurement,
        backend_controller=_Quel3Controller(),
        experiment_system=experiment_system,
    )

    result = asyncio.run(measurement.run_measurement(schedule=schedule, config=config))

    assert called["validated_schedule"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["experiment_system"] is experiment_system
    constraint_profile = cast(Any, called["constraint_profile"])
    assert constraint_profile.enforce_block_alignment is False
    assert result.device_config == {"kind": "quel3"}


def test_run_sweep_measurement_raises_not_implemented() -> None:
    """Given sweep measurement call, when invoked, then NotImplementedError is raised."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    with pytest.raises(NotImplementedError, match="run_sweep_measurement"):
        asyncio.run(measurement.run_sweep_measurement(config=SweepMeasurementConfig()))


def test_execute_async_entrypoint_is_removed() -> None:
    """Given measurement facade, execute_async entrypoint is not exposed."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )

    assert not hasattr(measurement, "execute_async")


def test_disconnect_delegates_to_session_service() -> None:
    """Given connected session service, disconnect delegates to session service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called = {"disconnect": 0}

    class _SessionService:
        def disconnect(self) -> None:
            called["disconnect"] += 1

    measurement.__dict__["_session_service"] = _SessionService()

    measurement.disconnect()

    assert called["disconnect"] == 1


def test_classifier_apis_delegate_to_classification_service() -> None:
    """Given classification API calls, when invoked, then they delegate to classification service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    classifiers = {"Q00": object()}
    confusion = np.array([[1.0]])
    inverse = np.array([[2.0]])
    called: dict[str, object] = {}

    class _ClassificationService:
        @property
        def classifiers(self) -> TargetMap:  # type: ignore[type-arg]
            return classifiers

        def update_classifiers(self, new_classifiers: TargetMap) -> None:  # type: ignore[type-arg]
            called["updated"] = new_classifiers

        def get_confusion_matrix(self, targets: list[str]) -> np.ndarray:
            called["confusion_targets"] = targets
            return confusion

        def get_inverse_confusion_matrix(self, targets: list[str]) -> np.ndarray:
            called["inverse_targets"] = targets
            return inverse

    measurement.__dict__["_classification_service"] = _ClassificationService()

    updated = cast(TargetMap, {"Q01": object()})  # type: ignore[type-arg]
    assert measurement.classifiers is classifiers
    measurement.update_classifiers(updated)
    assert called["updated"] is updated
    assert np.array_equal(measurement.get_confusion_matrix(["Q00"]), confusion)
    assert np.array_equal(
        measurement.get_inverse_confusion_matrix(["Q00"]),
        inverse,
    )
    assert called["confusion_targets"] == ["Q00"]
    assert called["inverse_targets"] == ["Q00"]


def test_apply_dc_voltages_delegates_to_amplification_service() -> None:
    """Given DC-voltage API call, when context is entered, then it delegates to amplification service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, object] = {}

    class _AmplificationService:
        @contextmanager
        def apply_dc_voltages(self, targets: str | list[str]):  # type: ignore[no-untyped-def]
            called["targets"] = targets
            called["entered"] = True
            try:
                yield
            finally:
                called["exited"] = True

    measurement.__dict__["_amplification_service"] = _AmplificationService()

    with measurement.apply_dc_voltages(["Q00"]):
        called["inside"] = True

    assert called["targets"] == ["Q00"]
    assert called["entered"] is True
    assert called["inside"] is True
    assert called["exited"] is True
