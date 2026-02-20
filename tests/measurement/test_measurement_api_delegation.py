"""Tests for legacy API delegation to schedule execution APIs."""

from __future__ import annotations

from types import MethodType
from typing import Any, ClassVar

import numpy as np
from qxpulse import PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1BackendRawResult
from qubex.measurement.measurement_client import MeasurementClient
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
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
from qubex.measurement.models.measurement_result import MeasurementResult


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
    measurement = MeasurementClient(
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
            "backend_controller": type("_BC", (), {"box_config": {"shots": 1}})(),
            "experiment_system": experiment_system,
        },
    )()
    measurement.__dict__["_system_manager"] = type(
        "_SM",
        (),
        {"rawdata_dir": None},
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
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
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


def test_measure_initializes_optional_flags_with_measure_defaults() -> None:
    """Given None optional flags, when measure is called, then it applies measure defaults."""
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    multiple = _make_multiple_result()
    called: dict[str, Any] = {}

    def fake_execute(
        self: MeasurementClient, **kwargs: object
    ) -> MultipleMeasureResult:
        called["kwargs"] = kwargs
        return multiple

    measurement.execute = MethodType(fake_execute, measurement)

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
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, Any] = {}
    expected = object()

    def fake_measure(self: MeasurementClient, **kwargs: object) -> Any:
        called["kwargs"] = kwargs
        return expected

    measurement.measure = MethodType(fake_measure, measurement)

    result = measurement.measure_noise(["Q00"], duration=1024.0)

    assert result is expected
    kwargs = called["kwargs"]
    assert kwargs["enable_dsp_sum"] is False
    assert kwargs["readout_duration"] == 1024.0
    assert kwargs["readout_amplitudes"] == {"Q00": 0}


def test_execute_initializes_optional_flags_with_execute_defaults() -> None:
    """Given None optional flags, when execute is called, then it applies execute defaults."""
    measurement = MeasurementClient(
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
        self: MeasurementClient,
        *,
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        called["build_kwargs"] = kwargs
        return built_schedule

    def fake_execute_measurement_schedule(
        self: MeasurementClient,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        called["config"] = config
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
            "backend_controller": type("_BC", (), {"box_config": {"shots": 1}})(),
            "experiment_system": experiment_system,
        },
    )()
    measurement.__dict__["_system_manager"] = type("_SM", (), {"rawdata_dir": None})()

    measurement.execute(
        schedule=pulse_schedule,
        add_pump_pulses=None,
        enable_dsp_demodulation=None,
        enable_dsp_classification=None,
        save_result=False,
    )

    assert called["build_kwargs"]["add_pump_pulses"] is False
    config = called["config"]
    assert config.dsp.enable_dsp_demodulation is True
    assert config.dsp.enable_dsp_sum is True
    assert config.dsp.enable_dsp_classification is False


def test_execute_measurement_schedule_delegates_to_executor(
    monkeypatch,
) -> None:
    """Given schedule execution inputs, when method is called, then it delegates to executor."""
    measurement = MeasurementClient(
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
        def execute(
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
    measurement.__dict__["_backend_manager"] = type(
        "_BM",
        (),
        {
            "backend_controller": backend_controller,
            "experiment_system": experiment_system,
        },
    )()

    monkeypatch.setattr(
        MeasurementScheduleExecutor,
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
    result = measurement.execute_measurement_schedule(schedule=schedule, config=config)

    assert called["schedule"] is schedule
    assert called["config"] is config
    assert result is expected


def test_execute_measurement_schedule_uses_backend_custom_factories(
    monkeypatch,
) -> None:
    """Given backend factory hooks, when executing a schedule, then MeasurementClient uses the custom path."""
    measurement = MeasurementClient(
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

    def _unexpected_backend_executor(**kwargs: object) -> object:
        raise AssertionError("Quel1BackendExecutor fallback should not be used.")

    def _unexpected_adapter(**kwargs: object) -> object:
        raise AssertionError(
            "Quel1MeasurementBackendAdapter fallback should not be used."
        )

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

    class _CustomBackendExecutor:
        def execute(self, *, request: BackendExecutionRequest) -> Quel1BackendRawResult:
            called["request"] = request
            return Quel1BackendRawResult(status={}, data={}, config={})

    class _CustomAdapter:
        sampling_period = 0.4
        constraint_profile = None

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

    class _CustomResultFactory:
        def create(self, **kwargs: object) -> MeasurementResult:
            called["result_kwargs"] = kwargs
            return MeasurementResult(
                mode="avg",
                data={"Q00": [np.array([1.0 + 0.0j])]},
                device_config={"kind": "quel3"},
                measurement_config={"mode": "avg"},
            )

    class _BackendController:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}
        DEFAULT_SAMPLING_PERIOD: ClassVar[float] = 0.4
        MEASUREMENT_CONSTRAINT_MODE: ClassVar[str] = "quel3"

        def create_measurement_backend_executor(
            self,
            *,
            execution_mode: str | None,
            clock_health_checks: bool | None,
        ) -> _CustomBackendExecutor:
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return _CustomBackendExecutor()

        def create_measurement_backend_adapter(
            self,
            *,
            experiment_system: object,
            constraint_profile: object,
        ) -> _CustomAdapter:
            called["experiment_system"] = experiment_system
            called["constraint_profile"] = constraint_profile
            return _CustomAdapter()

        def create_measurement_result_factory(
            self,
            *,
            experiment_system: object,
        ) -> _CustomResultFactory:
            called["result_factory_experiment_system"] = experiment_system
            return _CustomResultFactory()

    experiment_system = object()
    measurement.__dict__["_backend_manager"] = type(
        "_BM",
        (),
        {
            "backend_controller": _BackendController(),
            "experiment_system": experiment_system,
        },
    )()

    result = measurement.execute_measurement_schedule(schedule=schedule, config=config)

    assert called["validated_schedule"] is schedule
    assert called["request_schedule"] is schedule
    assert called["request_config"] is config
    assert called["experiment_system"] is experiment_system
    assert called["result_factory_experiment_system"] is experiment_system
    assert result.device_config == {"kind": "quel3"}


def test_disconnect_delegates_to_backend_manager() -> None:
    """Given connected manager, disconnect delegates to backend manager."""
    measurement = MeasurementClient(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called = {"disconnect": 0}

    class _BackendManager:
        def disconnect(self) -> None:
            called["disconnect"] += 1

    measurement.__dict__["_backend_manager"] = _BackendManager()

    measurement.disconnect()

    assert called["disconnect"] == 1
