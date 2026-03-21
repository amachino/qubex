"""Tests for legacy API delegation to schedule execution APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from qxpulse import PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1BackendExecutionResult
from qubex.measurement.measurement import Measurement
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.models import (
    CaptureData,
    MeasurementConfig,
    MeasurementSchedule,
    NDSweepMeasurementResult,
    Quel1MeasurementOptions,
    SweepMeasurementResult,
    SweepPoint,
    SweepValue,
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
from qubex.system import PortType
from qubex.typing import MeasurementMode, TargetMap


def _make_config(
    *,
    mode: MeasurementMode = "avg",
    shots: int = 2,
    time_integration: bool = False,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=time_integration,
        state_classification=False,
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


def _make_measurement_result(
    *,
    data: dict[str, list[np.ndarray]],
    measurement_config: MeasurementConfig,
    sampling_period: float,
    device_config: dict[str, object] | None = None,
) -> MeasurementResult:
    capture_data = {
        target: [
            CaptureData.from_primary_data(
                target=target,
                data=np.asarray(raw),
                config=measurement_config,
                sampling_period=sampling_period,
            )
            for raw in captures
        ]
        for target, captures in data.items()
    }
    return MeasurementResult(
        data=capture_data,
        measurement_config=measurement_config,
        device_config=device_config,
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


def test_execute_delegates_to_schedule_executor_with_built_schedule(
    monkeypatch,
) -> None:
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

    class _Executor:
        def execute_sync(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
            quel1_options: Quel1MeasurementOptions | None = None,
        ) -> MeasurementResult:
            _ = quel1_options
            called["run_schedule"] = schedule
            called["run_config"] = config
            return MeasurementResultConverter.from_multiple(
                multiple,
                measurement_config=_make_config(),
            )

    execution_service = measurement.execution_service
    execution_service.build_measurement_schedule = MethodType(
        fake_build, execution_service
    )
    monkeypatch.setattr(
        MeasurementExecutionService,
        "measurement_schedule_runner",
        property(lambda self: _Executor()),
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
        final_measurement=True,
    )

    assert result.mode == multiple.mode
    assert np.array_equal(
        result.data["Q00"][0].raw,
        multiple.data["Q00"][0].raw,
    )
    assert called["build_schedule"] is pulse_schedule
    assert called["run_schedule"] is built_schedule
    assert called["build_kwargs"]["final_measurement"] is True
    assert called["run_config"].shot_averaging is True


def test_execute_forwards_frequency_overrides_to_schedule_builder(
    monkeypatch,
) -> None:
    """Given execute frequency overrides, when execute is called, then schedule build receives frequencies."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    pulse_schedule = PulseSchedule(["Q00"])
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q00"]),
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

    class _Executor:
        def execute_sync(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
            quel1_options: Quel1MeasurementOptions | None = None,
        ) -> MeasurementResult:
            _ = (schedule, config, quel1_options)
            return MeasurementResultConverter.from_multiple(
                multiple,
                measurement_config=_make_config(),
            )

    execution_service = measurement.execution_service
    execution_service.build_measurement_schedule = MethodType(
        fake_build, execution_service
    )
    monkeypatch.setattr(
        MeasurementExecutionService,
        "measurement_schedule_runner",
        property(lambda self: _Executor()),
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

    _ = measurement.execute(
        schedule=pulse_schedule,
        frequencies={"Q00": 5.12},
    )

    assert called["build_schedule"] is pulse_schedule
    assert called["build_kwargs"]["frequencies"] == {"Q00": 5.12}


def test_capture_loopback_delegates_to_execution_service() -> None:
    """Given loopback capture inputs, when capture_loopback is called, then it delegates to execution service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    schedule = PulseSchedule(["Q00"])
    loopback_result = _make_measurement_result(
        data={"Q00": [np.array([1.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg", shots=128),
        sampling_period=2.0,
    )
    called: dict[str, object] = {}

    def fake_capture_loopback(
        self: MeasurementExecutionService,
        **kwargs: object,
    ) -> MeasurementResult:
        called["kwargs"] = kwargs
        return loopback_result

    measurement.execution_service.capture_loopback = MethodType(
        fake_capture_loopback,
        measurement.execution_service,
    )

    result = measurement.capture_loopback(schedule=schedule, n_shots=128)

    assert result is loopback_result
    kwargs = cast(dict[str, object], called["kwargs"])
    assert kwargs["schedule"] is schedule
    assert kwargs["n_shots"] == 128


def test_temporary_loopback_rfswitches_sets_and_restores_ports() -> None:
    """Given loopback capture targets, when capture_loopback runs, then rfswitches are applied and restored."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    monitor_out_port = SimpleNamespace(
        id="B0.MNTR0.OUT",
        box_id="B0",
        number=3,
        type=PortType.MNTR_OUT,
        rfswitch="pass",
    )
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(
        id="B0",
        ports=[read_out_port, read_in_port, monitor_out_port, monitor_in_port],
    )

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}
            self._port_by_number = {
                (port.box_id, port.number): port for port in box.ports
            }

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

        def set_port_params(
            self,
            box_id: str,
            port_number: int,
            *,
            rfswitch: str | None = None,
        ) -> None:
            if rfswitch is None:
                return
            self._port_by_number[(box_id, port_number)].rfswitch = rfswitch

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str | None]] = []

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            rfswitch: str | None = None,
        ) -> None:
            self.calls.append((box_name, port, rfswitch))

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    captured_build_kwargs: dict[str, Any] = {}

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule)
        captured_build_kwargs.update(kwargs)
        return built_schedule

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        assert read_in_port.rfswitch == "loop"
        assert read_out_port.rfswitch == "block"
        assert monitor_in_port.rfswitch == "loop"
        assert monitor_out_port.rfswitch == "pass"
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=_make_config(),
            sampling_period=2.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    _ = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]), n_shots=16)

    assert read_in_port.rfswitch == "open"
    assert read_out_port.rfswitch == "pass"
    assert monitor_in_port.rfswitch == "open"
    assert monitor_out_port.rfswitch == "pass"
    assert captured_build_kwargs["capture_targets"] == ["B0.READ0.IN", "B0.MNTR0.IN"]
    assert len(backend_controller.calls) == 6
    assert all(call[1] != 3 for call in backend_controller.calls)


def test_temporary_loopback_rfswitches_restores_ports_on_error() -> None:
    """Given a measurement error, when capture_loopback exits, then rfswitches are restored."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port])

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}
            self._port_by_number = {
                (port.box_id, port.number): port for port in box.ports
            }

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

        def set_port_params(
            self,
            box_id: str,
            port_number: int,
            *,
            rfswitch: str | None = None,
        ) -> None:
            if rfswitch is None:
                return
            self._port_by_number[(box_id, port_number)].rfswitch = rfswitch

    class _BackendControllerStub:
        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            rfswitch: str | None = None,
        ) -> None:
            _ = (box_name, port, rfswitch)

    control_system = _ControlSystemStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=_BackendControllerStub(),
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule, kwargs)
        return built_schedule

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        raise RuntimeError("test-error")

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    with pytest.raises(RuntimeError, match="test-error"):
        _ = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]))

    assert read_in_port.rfswitch == "open"
    assert read_out_port.rfswitch == "pass"


def test_capture_loopback_skips_ports_without_rfswitch() -> None:
    """Given no-rfswitch ports, when capture_loopback is called, then measurement still runs."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port])

    class NoRfSwitchError(Exception):
        pass

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

        def set_port_params(
            self,
            box_id: str,
            port_number: int,
            *,
            rfswitch: str | None = None,
        ) -> None:
            _ = (box_id, port_number, rfswitch)

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str | None]] = []

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            rfswitch: str | None = None,
        ) -> None:
            self.calls.append((box_name, port, rfswitch))
            raise NoRfSwitchError()

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule, kwargs)
        return built_schedule

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        assert read_in_port.rfswitch == "open"
        assert read_out_port.rfswitch == "pass"
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=_make_config(),
            sampling_period=2.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    result = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]))

    assert "Q00" in result.data
    assert read_in_port.rfswitch == "open"
    assert read_out_port.rfswitch == "pass"
    assert len(backend_controller.calls) == 2


def test_capture_loopback_initializes_awg_and_capunits_when_supported() -> None:
    """Given backend reset support, when capture_loopback runs, then AWG/CAP units are initialized."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port])

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}
            self._port_by_number = {
                (port.box_id, port.number): port for port in box.ports
            }

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

        def set_port_params(
            self,
            box_id: str,
            port_number: int,
            *,
            rfswitch: str | None = None,
        ) -> None:
            if rfswitch is None:
                return
            self._port_by_number[(box_id, port_number)].rfswitch = rfswitch

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.config_calls: list[tuple[str, int, str | None]] = []
            self.init_calls: list[list[str]] = []

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            rfswitch: str | None = None,
        ) -> None:
            self.config_calls.append((box_name, port, rfswitch))

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            self.init_calls.append(list(box_ids))

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    built_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule, kwargs)
        return built_schedule

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        assert backend_controller.init_calls == [["B0"]]
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=_make_config(),
            sampling_period=2.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    _ = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]), n_shots=16)

    assert backend_controller.init_calls == [["B0"]]


def test_capture_loopback_retries_with_read_in_only_after_e7_error() -> None:
    """Given broken capture data errors, when loopback capture retries, then it falls back to READ_IN only."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    monitor_out_port = SimpleNamespace(
        id="B0.MNTR0.OUT",
        box_id="B0",
        number=3,
        type=PortType.MNTR_OUT,
        rfswitch="pass",
    )
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(
        id="B0",
        ports=[read_out_port, read_in_port, monitor_out_port, monitor_in_port],
    )

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}
            self._port_by_number = {
                (port.box_id, port.number): port for port in box.ports
            }

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

        def set_port_params(
            self,
            box_id: str,
            port_number: int,
            *,
            rfswitch: str | None = None,
        ) -> None:
            if rfswitch is None:
                return
            self._port_by_number[(box_id, port_number)].rfswitch = rfswitch

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.config_calls: list[tuple[str, int, str | None]] = []
            self.init_calls: list[list[str]] = []

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            rfswitch: str | None = None,
        ) -> None:
            self.config_calls.append((box_name, port, rfswitch))

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            self.init_calls.append(list(box_ids))

    class E7awgCaptureDataError(Exception):
        pass

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        if label == "Q00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    capture_target_calls: list[list[str]] = []
    call_count = {"run": 0}

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule)
        capture_targets = cast(list[str], kwargs["capture_targets"])
        capture_target_calls.append(list(capture_targets))
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["Q00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        call_count["run"] += 1
        if call_count["run"] < 3:
            raise E7awgCaptureDataError()
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=_make_config(),
            sampling_period=2.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    result = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]))

    assert "Q00" in result.data
    assert call_count["run"] == 3
    assert capture_target_calls[0] == ["B0.READ0.IN", "B0.MNTR0.IN"]
    assert capture_target_calls[1] == ["B0.READ0.IN", "B0.MNTR0.IN"]
    assert capture_target_calls[2] == ["B0.READ0.IN"]


def test_capture_loopback_runs_without_dsp_demodulation() -> None:
    """Given loopback capture execution, when run_measurement is called, then QuEL-1 DSP demodulation is disabled."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    read_out_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        rfswitch="pass",
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port])

    class _ControlSystemStub:
        def __init__(self) -> None:
            self.boxes = [box]
            self._port_by_id = {port.id: port for port in box.ports}

        def get_port_by_id(self, port_id: str) -> Any:
            return self._port_by_id[port_id]

        def get_box(self, box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

    class _BackendControllerStub:
        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

    schedule_target = SimpleNamespace(
        label="Q00",
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0")),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _get_cap_target(label: str) -> Any:
        if label == "RQ00":
            return SimpleNamespace(channel=SimpleNamespace(port=read_in_port))
        raise KeyError(label)

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=_ControlSystemStub(),
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_cap_target=_get_cap_target,
    )
    _bind_runtime(
        measurement,
        backend_controller=_BackendControllerStub(),
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule, kwargs)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["Q00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    called: dict[str, object] = {}

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        called["config"] = config
        _ = (self, schedule)
        called["quel1_options"] = quel1_options
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=_make_config(),
            sampling_period=2.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    _ = measurement.capture_loopback(schedule=PulseSchedule(["Q00"]), n_shots=16)

    options = cast(Quel1MeasurementOptions | None, called["quel1_options"])
    config = cast(MeasurementConfig, called["config"])
    assert config.shot_averaging is False
    assert config.n_shots == 16
    assert options is not None
    assert options.demodulation is False


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

    assert called["kwargs"]["final_measurement"] is True
    assert called["kwargs"]["time_integration"] is False
    assert called["kwargs"]["plot"] is None
    assert result.data["Q00"] is multiple.data["Q00"][0]


def test_measure_accepts_deprecated_alias_options() -> None:
    """Given deprecated alias options, when measure is called, then it forwards them for compatibility."""
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
        enable_dsp_sum=None,
        enable_dsp_demodulation=None,
        enable_dsp_classification=None,
    )

    kwargs = called["kwargs"]
    assert kwargs["final_measurement"] is True
    assert kwargs["time_integration"] is False
    assert kwargs["add_pump_pulses"] is None
    assert kwargs["enable_dsp_sum"] is None
    assert kwargs["enable_dsp_demodulation"] is None
    assert kwargs["enable_dsp_classification"] is None
    assert kwargs["plot"] is None


def test_measure_noise_runs_via_run_measurement_with_noise_defaults() -> None:
    """Given noise measurement inputs, when measure_noise is called, then it builds and runs a noise schedule with explicit defaults."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, Any] = {}
    measurement_schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )
    measurement_config = _make_config(mode="avg", shots=1)

    def fake_create_measurement_config(
        self: MeasurementExecutionService,
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        _ = self
        called["config_kwargs"] = {
            "n_shots": n_shots,
            "shot_interval": shot_interval,
            "shot_averaging": shot_averaging,
            "time_integration": time_integration,
            "state_classification": state_classification,
        }
        return measurement_config

    def fake_build_measurement_schedule(
        self: MeasurementExecutionService,
        *,
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        _ = self
        called["pulse_schedule"] = pulse_schedule
        called["build_kwargs"] = kwargs
        return measurement_schedule

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = self
        called["run_schedule"] = schedule
        called["run_config"] = config
        called["run_quel1_options"] = quel1_options
        return _make_measurement_result(
            data={"Q00": [np.array([1.0 + 0.0j])]},
            measurement_config=measurement_config,
            sampling_period=2.0,
        )

    execution_service = measurement.execution_service
    execution_service.create_measurement_config = MethodType(
        fake_create_measurement_config,
        execution_service,
    )
    execution_service.build_measurement_schedule = MethodType(
        fake_build_measurement_schedule,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        fake_run_measurement,
        execution_service,
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

    result = asyncio.run(measurement.measure_noise(["Q00"], duration=1024.0))

    assert np.array_equal(result.data["Q00"][0].data, np.array([1.0 + 0.0j]))
    assert result.measurement_config is measurement_config
    config_kwargs = cast(dict[str, Any], called["config_kwargs"])
    assert config_kwargs["n_shots"] == 1
    assert config_kwargs["shot_averaging"] is True
    assert config_kwargs["time_integration"] is False
    assert config_kwargs["state_classification"] is False
    build_kwargs = cast(dict[str, Any], called["build_kwargs"])
    assert build_kwargs["readout_duration"] == 1024.0
    assert build_kwargs["readout_amplitudes"] == {"Q00": 0}
    assert build_kwargs["readout_amplification"] is False
    assert build_kwargs["final_measurement"] is True
    assert called["run_schedule"] is measurement_schedule
    assert called["run_config"] is measurement_config
    assert called["run_quel1_options"] is None


def test_execute_initializes_optional_flags_with_execute_defaults(
    monkeypatch,
) -> None:
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

    class _Executor:
        def execute_sync(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
            quel1_options: Quel1MeasurementOptions | None = None,
        ) -> MeasurementResult:
            _ = (schedule, quel1_options)
            called["config"] = config
            return MeasurementResultConverter.from_multiple(
                multiple,
                measurement_config=_make_config(),
            )

    execution_service = measurement.execution_service
    execution_service.build_measurement_schedule = MethodType(
        fake_build, execution_service
    )
    monkeypatch.setattr(
        MeasurementExecutionService,
        "measurement_schedule_runner",
        property(lambda self: _Executor()),
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
    )

    assert called["build_kwargs"]["readout_amplification"] is False
    assert called["build_kwargs"]["plot"] is False
    config = called["config"]
    assert config.time_integration is True
    assert config.state_classification is False


def test_run_measurement_delegates_to_executor(
    monkeypatch,
) -> None:
    """Given schedule execution inputs, when method is called, then it delegates to executor with explicit options."""
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
    expected = MeasurementResultConverter.from_multiple(
        _make_multiple_result(),
        measurement_config=_make_config(),
    )
    called: dict[str, Any] = {}

    class _Executor:
        async def execute_async(self, **kwargs: Any) -> MeasurementResult:
            called["schedule"] = kwargs["schedule"]
            called["config"] = kwargs["config"]
            called["has_quel1_options"] = "quel1_options" in kwargs
            called["quel1_options"] = kwargs.get("quel1_options")
            return expected

    experiment_system = type("_ES", (), {})()
    backend_controller = type("_BC", (), {})()
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    monkeypatch.setattr(
        MeasurementExecutionService,
        "measurement_schedule_runner",
        property(lambda self: _Executor()),
    )
    result = asyncio.run(measurement.run_measurement(schedule=schedule, config=config))

    assert called["schedule"] is schedule
    assert called["config"] is config
    assert called["has_quel1_options"] is True
    assert called["quel1_options"] is None
    assert result is expected


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
            sampling_period: float,
        ) -> MeasurementResult:
            called["result_kwargs"] = {
                "backend_result": backend_result,
                "measurement_config": measurement_config,
                "device_config": device_config,
                "sampling_period": sampling_period,
            }
            return _make_measurement_result(
                data={"Q00": [np.array([1.0 + 0.0j])]},
                measurement_config=_make_config(mode="avg"),
                device_config={"kind": "quel3"},
                sampling_period=0.4,
            )

    monkeypatch.setattr(
        "qubex.measurement.measurement_schedule_runner.Quel3MeasurementBackendAdapter",
        _Quel3Adapter,
    )

    class _Quel3Controller:
        box_config: ClassVar[dict[str, str]] = {"kind": "quel3"}
        sampling_period_ns: ClassVar[float] = 0.4
        CAPTURE_DECIMATION_FACTOR: ClassVar[int] = 4
        target_alias_map: ClassVar[dict[str, str]] = {}

        async def execute_async(
            self,
            *,
            request: BackendExecutionRequest,
            execution_mode: str | None = None,
            clock_health_checks: bool | None = None,
        ) -> Quel1BackendExecutionResult:
            called["request"] = request
            called["execution_mode"] = execution_mode
            called["clock_health_checks"] = clock_health_checks
            return Quel1BackendExecutionResult(status={}, data={}, config={})

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


def test_run_sweep_measurement_delegates_to_execution_service() -> None:
    """Given sweep measurement inputs, when invoked, then it delegates to execution service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    config = _make_config()
    sweep_values: list[SweepValue] = [0.1, 0.2]
    expected = SweepMeasurementResult(
        sweep_values=sweep_values,
        config=config,
        results=[],
    )
    called: dict[str, Any] = {}

    def schedule(point: SweepValue) -> MeasurementSchedule:
        del point
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["RQ00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_sweep_measurement(
        self: MeasurementExecutionService,
        schedule: Any,
        *,
        sweep_values: Any,
        config: MeasurementConfig | None,
        on_point: Any = None,
    ) -> SweepMeasurementResult:
        called["schedule"] = schedule
        called["sweep_values"] = sweep_values
        called["config"] = config
        called["on_point"] = on_point
        return expected

    measurement.execution_service.run_sweep_measurement = MethodType(
        fake_run_sweep_measurement,
        measurement.execution_service,
    )

    result = asyncio.run(
        measurement.run_sweep_measurement(
            schedule,
            sweep_values=sweep_values,
            config=config,
        )
    )

    assert called["schedule"] is schedule
    assert called["sweep_values"] is sweep_values
    assert called["config"] is config
    assert called["on_point"] is None
    assert result is expected


def test_run_sweep_measurement_calls_on_point_for_each_result() -> None:
    """Given on_point callback, when sweep runs, then callback receives each point result in order."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_values: list[SweepValue] = [0, 1]
    callbacks: list[tuple[SweepValue, MeasurementResult]] = []

    def schedule(point: SweepValue) -> MeasurementSchedule:
        step = int(point)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={"Q00": [np.array([step + 0.0j])]},
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_sweep_measurement(
            schedule,
            sweep_values=sweep_values,
            config=config,
            on_point=lambda value, measured: callbacks.append((value, measured)),
        )
    )

    assert [value for value, _ in callbacks] == sweep_values
    assert len(callbacks) == len(result.results)
    assert all(
        measured is expected
        for (_, measured), expected in zip(callbacks, result.results, strict=True)
    )


def test_run_sweep_measurement_runs_points_and_returns_results() -> None:
    """Given sweep points, when execution succeeds, then pointwise results are returned."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_values: list[SweepValue] = [0, 1]

    def schedule(point: SweepValue) -> MeasurementSchedule:
        step = int(point)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={
                "Q00": [
                    np.array([step + 1.0 + 0.0j]),
                    np.array([step + 11.0 + 0.0j]),
                ]
            },
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_sweep_measurement(
            schedule,
            sweep_values=sweep_values,
            config=config,
        )
    )

    assert result.sweep_values == sweep_values
    assert result.config == config
    assert np.array_equal(result.results[0].data["Q00"][0].data, np.array([1.0 + 0.0j]))
    assert np.array_equal(result.results[1].data["Q00"][0].data, np.array([2.0 + 0.0j]))
    assert result.results[0].measurement_config == config
    assert result.results[1].measurement_config == config


def test_run_sweep_measurement_data_property_returns_pointwise_data() -> None:
    """Given sweep results, when reading data property, then target-keyed sweep arrays are returned."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()

    def schedule(point: SweepValue) -> MeasurementSchedule:
        step = int(point)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={
                "Q00": [
                    np.array([step + 1.0 + 0.0j]),
                    np.array([step + 11.0 + 0.0j]),
                ]
            },
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_sweep_measurement(
            schedule,
            sweep_values=[0, 1],
            config=config,
        )
    )

    assert set(result.data) == {"Q00"}
    assert len(result.data["Q00"]) == 2
    assert np.array_equal(
        result.data["Q00"][0],
        np.array(
            [
                [1.0 + 0.0j],
                [2.0 + 0.0j],
            ]
        ),
    )
    assert np.array_equal(
        result.data["Q00"][1],
        np.array(
            [
                [11.0 + 0.0j],
                [12.0 + 0.0j],
            ]
        ),
    )


def test_run_sweep_measurement_data_property_uses_canonical_iq_series_shape() -> None:
    """Given time-integrated single-shot data, sweep arrays should expose one IQ value per shot."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config(mode="single", shots=2, time_integration=True)

    def schedule(point: SweepValue) -> MeasurementSchedule:
        step = int(point)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={
                "Q00": [
                    np.array(
                        [
                            [step + 1.0 + 0.0j],
                            [step + 2.0 + 0.0j],
                        ]
                    )
                ]
            },
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_sweep_measurement(
            schedule,
            sweep_values=[0, 1],
            config=config,
        )
    )

    assert result.data["Q00"][0].shape == (2, 2)
    assert np.array_equal(
        result.data["Q00"][0],
        np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j],
                [2.0 + 0.0j, 3.0 + 0.0j],
            ]
        ),
    )


def test_run_sweep_measurement_resolves_default_config() -> None:
    """Given omitted sweep config, when running, then default config is resolved once and reused."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    default_config = _make_config()
    sweep_values: list[SweepValue] = [0]
    called: dict[str, object] = {}

    def schedule(point: SweepValue) -> MeasurementSchedule:
        del point
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["RQ00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    def fake_create_measurement_config(
        self: MeasurementExecutionService,
        **_: object,
    ) -> MeasurementConfig:
        del self
        called["create_called"] = True
        return default_config

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self, schedule
        called["config"] = config
        return _make_measurement_result(
            data={"Q00": [np.array([0.0 + 0.0j])]},
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.create_measurement_config = MethodType(  # type: ignore[method-assign]
        fake_create_measurement_config, execution_service
    )
    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_sweep_measurement(
            schedule,
            sweep_values=sweep_values,
            config=None,
        )
    )

    assert called["create_called"] is True
    assert called["config"] is default_config
    assert result.config is default_config


def test_run_sweep_measurement_stops_immediately_on_error() -> None:
    """Given pointwise execution error, when running sweep, then it fails fast."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_values: list[SweepValue] = [0, 1, 2]
    called: dict[str, int] = {"count": 0}

    def schedule(point: SweepValue) -> MeasurementSchedule:
        step = int(point)
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self, schedule, config
        called["count"] += 1
        if called["count"] == 2:
            raise RuntimeError("boom")
        return _make_measurement_result(
            data={"Q00": [np.array([0.0 + 0.0j])]},
            measurement_config=_make_config(mode="avg"),
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            execution_service.run_sweep_measurement(
                schedule,
                sweep_values=sweep_values,
                config=config,
            )
        )

    assert called["count"] == 2


def test_run_ndsweep_measurement_delegates_to_execution_service() -> None:
    """Given ndsweep measurement inputs, when invoked, then it delegates to execution service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    config = _make_config()
    sweep_points: dict[str, Sequence[SweepValue]] = {
        "amp": [0.1, 0.2],
        "step": [0, 1],
    }
    sweep_axes = ("amp", "step")
    expected = NDSweepMeasurementResult(
        sweep_points={"amp": [0.1, 0.2], "step": [0, 1]},
        sweep_axes=sweep_axes,
        shape=(2, 2),
        config=config,
        results=[],
    )
    called: dict[str, object] = {}

    def schedule(point: SweepPoint) -> MeasurementSchedule:
        del point
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["RQ00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_ndsweep_measurement(
        self: MeasurementExecutionService,
        schedule: Any,
        *,
        sweep_points: Any,
        sweep_axes: Any,
        config: MeasurementConfig | None,
    ) -> NDSweepMeasurementResult:
        del self
        called["schedule"] = schedule
        called["sweep_points"] = sweep_points
        called["sweep_axes"] = sweep_axes
        called["config"] = config
        return expected

    measurement.execution_service.run_ndsweep_measurement = MethodType(
        fake_run_ndsweep_measurement,
        measurement.execution_service,
    )

    result = asyncio.run(
        measurement.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            sweep_axes=sweep_axes,
            config=config,
        )
    )

    assert called["schedule"] is schedule
    assert called["sweep_points"] is sweep_points
    assert called["sweep_axes"] == sweep_axes
    assert called["config"] is config
    assert result is expected


def test_run_ndsweep_measurement_runs_cartesian_order_and_helpers() -> None:
    """Given ndsweep inputs, when execution succeeds, then C-order Cartesian results and helper accessors work."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_points: dict[str, Sequence[SweepValue]] = {
        "amp": [0.1, 0.2],
        "step": [0, 1, 2],
    }
    sweep_axes = ("amp", "step")
    scheduled_points: list[SweepPoint] = []

    def schedule(point: SweepPoint) -> MeasurementSchedule:
        scheduled_points.append(point)
        step = int(point["step"])
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={"Q00": [np.array([step + 1.0 + 0.0j])]},
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            sweep_axes=sweep_axes,
            config=config,
        )
    )

    assert result.shape == (2, 3)
    assert result.sweep_axes == sweep_axes
    assert len(result.results) == 6
    assert scheduled_points == [
        {"amp": 0.1, "step": 0},
        {"amp": 0.1, "step": 1},
        {"amp": 0.1, "step": 2},
        {"amp": 0.2, "step": 0},
        {"amp": 0.2, "step": 1},
        {"amp": 0.2, "step": 2},
    ]
    assert np.array_equal(
        result.get((1, 2)).data["Q00"][0].data,
        np.array([3.0 + 0.0j]),
    )
    assert result.get((1, 2)) is result.results[5]
    assert result.get_sweep_point((1, 0)) == {"amp": 0.2, "step": 0}
    assert result.get_sweep_point((1, 1)) == {"amp": 0.2, "step": 1}
    with pytest.raises(TypeError):
        _ = result.get(5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = result.get_sweep_point(4)  # type: ignore[arg-type]
    assert all(item.measurement_config == config for item in result.results)


def test_run_ndsweep_measurement_data_property_returns_flattened_pointwise_data() -> (
    None
):
    """Given ndsweep results, when reading data property, then target-keyed flattened sweep arrays are returned."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_points: dict[str, Sequence[SweepValue]] = {
        "amp": [0.1, 0.2],
        "step": [0, 1],
    }

    def schedule(point: SweepPoint) -> MeasurementSchedule:
        step = int(point["step"])
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={
                "Q00": [
                    np.array([step + 1.0 + 0.0j]),
                    np.array([step + 11.0 + 0.0j]),
                ]
            },
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            sweep_axes=("amp", "step"),
            config=config,
        )
    )

    assert set(result.data) == {"Q00"}
    assert len(result.data["Q00"]) == 2
    assert result.data["Q00"][0].shape == (2, 2, 1)
    assert result.data["Q00"][1].shape == (2, 2, 1)
    assert np.array_equal(
        result.data["Q00"][0],
        np.array(
            [
                [[1.0 + 0.0j], [2.0 + 0.0j]],
                [[1.0 + 0.0j], [2.0 + 0.0j]],
            ]
        ),
    )
    assert np.array_equal(
        result.data["Q00"][1],
        np.array(
            [
                [[11.0 + 0.0j], [12.0 + 0.0j]],
                [[11.0 + 0.0j], [12.0 + 0.0j]],
            ]
        ),
    )


def test_run_ndsweep_measurement_data_property_uses_canonical_iq_series_shape() -> None:
    """Given time-integrated sweep data, ND arrays should expose one IQ value per shot."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config(mode="single", shots=2, time_integration=True)
    sweep_points: dict[str, Sequence[SweepValue]] = {
        "amp": [0.1, 0.2],
        "step": [0, 1],
    }

    def schedule(point: SweepPoint) -> MeasurementSchedule:
        step = int(point["step"])
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule([f"RQ0{step}"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self
        step = int(schedule.pulse_schedule.labels[0][-1])
        return _make_measurement_result(
            data={
                "Q00": [
                    np.array(
                        [
                            [step + 1.0 + 0.0j],
                            [step + 2.0 + 0.0j],
                        ]
                    )
                ]
            },
            measurement_config=config,
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            sweep_axes=("amp", "step"),
            config=config,
        )
    )

    assert result.data["Q00"][0].shape == (2, 2, 2)
    assert np.array_equal(
        result.data["Q00"][0],
        np.array(
            [
                [[1.0 + 0.0j, 2.0 + 0.0j], [2.0 + 0.0j, 3.0 + 0.0j]],
                [[1.0 + 0.0j, 2.0 + 0.0j], [2.0 + 0.0j, 3.0 + 0.0j]],
            ]
        ),
    )


def test_run_ndsweep_measurement_uses_input_axis_order_by_default() -> None:
    """Given omitted ndsweep axes, when running, then dict insertion order is used."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()
    sweep_points: dict[str, Sequence[SweepValue]] = {"z": [10, 20], "x": [1]}

    def schedule(point: SweepPoint) -> MeasurementSchedule:
        del point
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["RQ00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def fake_run_measurement(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        del self, schedule, config
        return _make_measurement_result(
            data={"Q00": [np.array([0.0 + 0.0j])]},
            measurement_config=_make_config(mode="avg"),
            sampling_period=2.0,
        )

    execution_service.run_measurement = MethodType(
        fake_run_measurement, execution_service
    )

    result = asyncio.run(
        execution_service.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            config=config,
        )
    )

    assert result.sweep_axes == ("z", "x")
    assert result.shape == (2, 1)


def test_run_ndsweep_measurement_requires_explicit_axes_for_non_dict_mapping() -> None:
    """Given non-dict sweep_points, when axes are omitted, then the call fails fast."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    execution_service = measurement.execution_service
    config = _make_config()

    class _SweepPoints(Mapping[str, Sequence[SweepValue]]):
        def __init__(self, data: dict[str, Sequence[SweepValue]]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Sequence[SweepValue]:
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    sweep_points: Mapping[str, Sequence[SweepValue]] = _SweepPoints(
        {"z": [10, 20], "x": [1]}
    )

    def schedule(point: Mapping[str, SweepValue]) -> MeasurementSchedule:
        del point
        return MeasurementSchedule(
            pulse_schedule=PulseSchedule(["RQ00"]),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    with pytest.raises(
        ValueError,
        match=(
            r"sweep_axes must be provided when sweep_points is not a "
            r"dict-derived insertion-ordered mapping\."
        ),
    ):
        asyncio.run(
            execution_service.run_ndsweep_measurement(
                schedule,
                sweep_points=sweep_points,
                config=config,
            )
        )


def test_disconnect_delegates_to_session_service() -> None:
    """Given connected session service, when disconnect is called, then disconnect is delegated to the session service."""
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
