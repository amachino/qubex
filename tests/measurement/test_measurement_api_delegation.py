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
    MeasurementStabilitySnapshot,
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
from qubex.measurement.services.measurement_monitor_service import (
    MeasurementMonitorService,
    _LoopbackMonitorSourceSetting,
)
from qubex.system import PortType
from qubex.system.quel1.quel1_system_constants import NCO_STEP_HZ
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
    measurement.monitor_service.__dict__["_context"] = context
    measurement.monitor_service.__dict__["_session_service"] = session_service


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


def test_execute_saves_raw_measurement_result_when_rawdata_dir_is_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Given rawdata_dir, when execute runs, then raw MeasurementResult is saved."""
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
    measurement_config = _make_config(mode="avg", shots=2)
    raw_result = _make_measurement_result(
        data={"Q00": [np.array([1.0 + 0.0j])]},
        measurement_config=measurement_config,
        sampling_period=2.0,
        device_config={"backend": "stub"},
    )

    def fake_build(
        self: MeasurementExecutionService,
        *,
        pulse_schedule: PulseSchedule,
        **kwargs: object,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule, kwargs)
        return built_schedule

    class _Executor:
        def execute_sync(
            self,
            *,
            schedule: MeasurementSchedule,
            config: MeasurementConfig,
            quel1_options: Quel1MeasurementOptions | None = None,
        ) -> MeasurementResult:
            _ = (self, schedule, config, quel1_options)
            return raw_result

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
    backend_controller = type("_BC", (), {"box_config": {"shots": 2}})()
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
        rawdata_dir=tmp_path,
    )

    converted = measurement.execute(schedule=pulse_schedule)

    saved_files = list(tmp_path.glob("*.nc"))
    assert len(saved_files) == 1
    restored = MeasurementResult.load(saved_files[0])
    assert np.array_equal(restored.data["Q00"][0].data, np.array([1.0 + 0.0j]))
    assert restored.device_config == {"backend": "stub"}
    assert np.array_equal(converted.data["Q00"][0].raw, np.array([1.0 + 0.0j]))


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


def test_capture_loopback_delegates_to_monitor_service() -> None:
    """Given loopback inputs, when capture_loopback is called, then it delegates to monitor service."""
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
        self: MeasurementMonitorService,
        **kwargs: object,
    ) -> MeasurementResult:
        called["kwargs"] = kwargs
        return loopback_result

    measurement.monitor_service.capture_loopback = MethodType(
        fake_capture_loopback,
        measurement.monitor_service,
    )

    result = measurement.capture_loopback(
        schedule=schedule,
        n_shots=128,
        block_outputs=False,
    )

    assert result is loopback_result
    kwargs = cast(dict[str, object], called["kwargs"])
    assert kwargs["schedule"] is schedule
    assert kwargs["n_shots"] == 128
    assert kwargs["block_outputs"] is False
    assert kwargs["shot_averaging"] is True
    assert kwargs["demodulation"] is True
    assert kwargs["include_read_in"] is False
    assert kwargs["configure_monitor_nco"] is True


def test_measurement_stability_methods_delegate_to_stability_service() -> None:
    """Given stability API calls, when invoked, then they delegate to the stability service."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: list[tuple[str, dict[str, object]]] = []
    baseline_result = object()
    update_result = object()

    def fake_establish(**kwargs: object) -> object:
        called.append(("establish", kwargs))
        return baseline_result

    def fake_update(**kwargs: object) -> object:
        called.append(("update", kwargs))
        return update_result

    cast(
        Any, measurement.stability_service
    ).establish_output_signal_baseline = fake_establish
    cast(
        Any, measurement.stability_service
    ).update_output_signal_corrections = fake_update

    baseline = measurement.establish_stability_baseline(
        targets=["Q00"],
        n_shots=128,
        trim_samples=4,
        estimate_gain_noise=False,
        estimate_phase_noise=False,
    )
    update = measurement.update_stability_corrections(
        targets=["RQ00"],
        max_gain_relative_step=0.002,
        gain_correction_deadband=0.003,
        auto_gain_correction_deadband=False,
        gain_correction_deadband_sigma=4.0,
        max_phase_step=0.04,
        phase_smoothing=0.2,
        phase_correction_deadband=0.01,
        auto_phase_correction_deadband=False,
        phase_correction_deadband_sigma=4.0,
        phase_min_resultant_length=0.8,
        trim_samples=5,
    )

    assert baseline is baseline_result
    assert update is update_result
    assert called[0][0] == "establish"
    assert "capture" not in called[0][1]
    assert called[0][1]["targets"] == ["Q00"]
    assert called[0][1]["n_shots"] == 128
    assert called[0][1]["trim_samples"] == 4
    assert called[0][1]["reference_scope"] == "box"
    assert called[0][1]["estimate_gain_noise"] is False
    assert called[0][1]["estimate_phase_noise"] is False
    assert called[1][0] == "update"
    assert "capture" not in called[1][1]
    assert called[1][1]["targets"] == ["RQ00"]
    assert called[1][1]["max_gain_relative_step"] == 0.002
    assert called[1][1]["gain_correction_deadband"] == 0.003
    assert called[1][1]["auto_gain_correction_deadband"] is False
    assert called[1][1]["gain_correction_deadband_sigma"] == 4.0
    assert called[1][1]["max_phase_step"] == 0.04
    assert called[1][1]["phase_smoothing"] == 0.2
    assert called[1][1]["phase_correction_deadband"] == 0.01
    assert called[1][1]["auto_phase_correction_deadband"] is False
    assert called[1][1]["phase_correction_deadband_sigma"] == 4.0
    assert called[1][1]["phase_min_resultant_length"] == 0.8
    assert called[1][1]["trim_samples"] == 5
    assert called[1][1]["reference_scope"] is None


def test_measurement_check_signal_stability_returns_snapshots() -> None:
    """Given signal stability API, when invoked, then it returns snapshots."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    called: dict[str, object] = {}
    snapshots = [MeasurementStabilitySnapshot(output_corrections={})]

    def fake_check_signal_stability(
        **kwargs: object,
    ) -> list[MeasurementStabilitySnapshot]:
        called.update(kwargs)
        return snapshots

    cast(
        Any, measurement.stability_service
    ).check_signal_stability = fake_check_signal_stability

    result = measurement.check_signal_stability(
        targets=["Q00"],
        duration=12.0,
        sample_interval=2.0,
        n_shots=128,
        trim_samples=4,
        gain_correction_deadband=0.003,
        auto_gain_correction_deadband=False,
        gain_correction_deadband_sigma=4.0,
        max_phase_step=0.04,
        phase_smoothing=0.2,
        phase_correction_deadband=0.01,
        auto_phase_correction_deadband=False,
        phase_correction_deadband_sigma=4.0,
        phase_min_resultant_length=0.8,
        update_corrections=True,
        plot=True,
    )

    assert result is snapshots
    assert "capture" not in called
    assert called["targets"] == ["Q00"]
    assert called["duration"] == 12.0
    assert called["sample_interval"] == 2.0
    assert called["n_shots"] == 128
    assert called["trim_samples"] == 4
    assert called["gain_correction_deadband"] == 0.003
    assert called["auto_gain_correction_deadband"] is False
    assert called["gain_correction_deadband_sigma"] == 4.0
    assert called["max_phase_step"] == 0.04
    assert called["phase_smoothing"] == 0.2
    assert called["phase_correction_deadband"] == 0.01
    assert called["auto_phase_correction_deadband"] is False
    assert called["phase_correction_deadband_sigma"] == 4.0
    assert called["phase_min_resultant_length"] == 0.8
    assert called["plot"] is True
    assert called["reference_scope"] == "box"
    assert called["update_corrections"] is True


@pytest.mark.parametrize(
    ("block_outputs", "expected_output_rfswitch", "expected_calls"),
    [
        (
            True,
            "block",
            [
                ("B0", 1, "block"),
                ("B0", 4, "loop"),
                ("B0", 1, "pass"),
                ("B0", 4, "open"),
            ],
        ),
        (
            False,
            "pass",
            [
                ("B0", 4, "loop"),
                ("B0", 4, "open"),
            ],
        ),
    ],
)
def test_capture_loopback_block_outputs_controls_active_outputs_and_restores_ports(
    block_outputs: bool,
    expected_output_rfswitch: str,
    expected_calls: list[tuple[str, int, str | None]],
) -> None:
    """Given loopback capture, when block_outputs is set, then active outputs follow it."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    output_port = SimpleNamespace(
        id="B0.CTRL0.OUT",
        box_id="B0",
        number=1,
        type=PortType.CTRL,
        rfswitch="pass",
    )
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        rfswitch="open",
    )
    box = SimpleNamespace(id="B0", ports=[output_port, monitor_in_port])

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
        channel=SimpleNamespace(port=output_port),
    )

    def _resolve_qubit_label(label: str) -> str:
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=_resolve_qubit_label,
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
        assert output_port.rfswitch == expected_output_rfswitch
        assert monitor_in_port.rfswitch == "loop"
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j])]},
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

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["Q00"]),
        n_shots=16,
        block_outputs=block_outputs,
    )

    assert output_port.rfswitch == "pass"
    assert monitor_in_port.rfswitch == "open"
    assert captured_build_kwargs["capture_targets"] == ["B0.MNTR0.IN"]
    assert backend_controller.calls == expected_calls


def test_capture_loopback_syncs_monitor_nco_to_active_source() -> None:
    """Given monitor loopback capture, when running, then monitor NCO follows the source output."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel = SimpleNamespace(number=0, fnco_freq=125_000_000)
    output_port = SimpleNamespace(
        id="B0.CTRL0",
        box_id="B0",
        number=1,
        type=PortType.CTRL,
        lo_freq=8_500_000_000,
        cnco_freq=1_617_187_500,
        channels=(source_channel,),
        rfswitch="pass",
    )
    source_channel.port = output_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=9_000_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(id="B0", ports=[output_port, monitor_in_port])

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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 4:
                return {1}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if (
                lo_freq_hz is not None
                or cnco_freq_hz is not None
                or cnco_locked_with is not None
            ):
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
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

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config)
        options = cast(Quel1MeasurementOptions, quel1_options)
        assert options.demodulation is False
        assert monitor_in_port.lo_freq == output_port.lo_freq
        assert monitor_in_port.cnco_freq == output_port.cnco_freq
        assert monitor_channel.fnco_freq == source_channel.fnco_freq
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j])]},
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

    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": 8_500_000_000,
            "cnco_freq_hz": None,
            "cnco_locked_with": 1,
        }
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 125_000_000,
        }
    ]

    backend_controller.port_config_calls.clear()
    backend_controller.runit_config_calls.clear()

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["Q00"]),
        n_shots=16,
        configure_monitor_nco=False,
    )

    assert backend_controller.port_config_calls == []
    assert backend_controller.runit_config_calls == []


def test_capture_loopback_preserves_type_a_monitor_lo_shared_with_pump() -> None:
    """Given Type A monitor LO shares PUMP LO, monitor capture preserves LO."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel = SimpleNamespace(number=0, fnco_freq=125_000_000)
    output_port = SimpleNamespace(
        id="B0.CTRL2",
        box_id="B0",
        number=5,
        type=PortType.CTRL,
        sideband="L",
        lo_freq=8_500_000_000,
        cnco_freq=1_617_187_500,
        channels=(source_channel,),
        rfswitch="pass",
    )
    source_channel.port = output_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=8_600_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="qube-riken-a"),
        ports=[output_port, monitor_in_port],
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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 4:
                return {5}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if (
                lo_freq_hz is not None
                or cnco_freq_hz is not None
                or cnco_locked_with is not None
            ):
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    expected_cnco_hz = 79 * NCO_STEP_HZ

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

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config)
        options = cast(Quel1MeasurementOptions, quel1_options)
        assert options.demodulation is False
        assert output_port.lo_freq == 8_500_000_000
        assert monitor_in_port.lo_freq == 8_600_000_000
        assert monitor_in_port.cnco_freq == expected_cnco_hz
        assert monitor_channel.fnco_freq == 0
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j])]},
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

    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": None,
            "cnco_freq_hz": expected_cnco_hz,
            "cnco_locked_with": None,
        }
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 0,
        }
    ]


def test_capture_loopback_preserves_type_b_monitor_lo_shared_with_ctrl() -> None:
    """Given Type B monitor LO shares CTRL LO, monitor capture preserves LO."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel = SimpleNamespace(number=0, fnco_freq=125_000_000)
    output_port = SimpleNamespace(
        id="B0.CTRL1",
        box_id="B0",
        number=2,
        type=PortType.CTRL,
        sideband="L",
        lo_freq=8_500_000_000,
        cnco_freq=1_617_187_500,
        channels=(source_channel,),
        rfswitch="pass",
    )
    source_channel.port = output_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=8_600_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="qube-riken-b"),
        ports=[output_port, monitor_in_port],
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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 4:
                return {2}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if (
                lo_freq_hz is not None
                or cnco_freq_hz is not None
                or cnco_locked_with is not None
            ):
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    expected_cnco_hz = 79 * NCO_STEP_HZ

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

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config)
        options = cast(Quel1MeasurementOptions, quel1_options)
        assert options.demodulation is False
        assert output_port.lo_freq == 8_500_000_000
        assert monitor_in_port.lo_freq == 8_600_000_000
        assert monitor_in_port.cnco_freq == expected_cnco_hz
        assert monitor_channel.fnco_freq == 0
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j])]},
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

    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": None,
            "cnco_freq_hz": expected_cnco_hz,
            "cnco_locked_with": None,
        }
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 0,
        }
    ]


def test_capture_loopback_configures_quel1se_r8_monitor_for_lo_less_source() -> None:
    """Given a `quel1se-riken8` LO-less source, when capturing monitor loopback, then use fixed monitor LO."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel = SimpleNamespace(number=0, fnco_freq=93_750_000)
    output_port = SimpleNamespace(
        id="B0.CTRL1",
        box_id="B0",
        number=7,
        type=PortType.CTRL,
        lo_freq=None,
        cnco_freq=4_265_625_000,
        channels=(source_channel,),
        rfswitch="pass",
    )
    source_channel.port = output_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR1.IN",
        box_id="B0",
        number=10,
        type=PortType.MNTR_IN,
        lo_freq=9_000_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="quel1se-riken8"),
        ports=[output_port, monitor_in_port],
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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 10:
                return {7}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if lo_freq_hz is not None or cnco_locked_with is not None:
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
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

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, config, quel1_options)
        assert monitor_in_port.lo_freq == 6_000_000_000
        assert monitor_in_port.cnco_freq == 1_640_625_000
        assert monitor_in_port.cnco_freq % NCO_STEP_HZ == 0
        assert monitor_channel.fnco_freq == 0
        return _make_measurement_result(
            data={"B0.MNTR1.IN": [np.array([1.0 + 0.0j])]},
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

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["Q00"]),
        n_shots=16,
        capture_targets=["B0.MNTR1.IN"],
        demodulation=False,
    )

    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 10,
            "lo_freq_hz": 6_000_000_000,
            "cnco_freq_hz": 1_640_625_000,
            "cnco_locked_with": None,
        }
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 10,
            "runit": 0,
            "fnco_freq_hz": 0,
        }
    ]


def test_capture_loopback_configures_quel1se_r8_monitor_for_readout_source() -> None:
    """Given a `quel1se-riken8` readout source, when capturing monitor loopback, then use USB with fixed monitor LO."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel = SimpleNamespace(number=0, fnco_freq=100_000_000)
    readout_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        sideband="L",
        lo_freq=8_500_000_000,
        cnco_freq=2_203_125_000,
        channels=(source_channel,),
        rfswitch="pass",
    )
    source_channel.port = readout_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=9_000_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="quel1se-riken8"),
        ports=[readout_port, monitor_in_port],
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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 4:
                return {1}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if lo_freq_hz is not None or cnco_locked_with is not None:
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    schedule_target = SimpleNamespace(
        label="RQ00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
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
            pulse_schedule=PulseSchedule(["RQ00"]),
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
        assert monitor_in_port.lo_freq == 6_000_000_000
        assert monitor_in_port.cnco_freq == 8 * NCO_STEP_HZ
        assert monitor_channel.fnco_freq == 0
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j])]},
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

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        n_shots=16,
        capture_targets=["B0.MNTR0.IN"],
        demodulation=False,
    )

    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": 6_000_000_000,
            "cnco_freq_hz": 8 * NCO_STEP_HZ,
            "cnco_locked_with": None,
        }
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 0,
        }
    ]


def test_capture_loopback_splits_monitor_runs_by_active_source_channel() -> None:
    """Given multiple monitor sources, when capturing, then each source gets a matched NCO run."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00", "Q01"],
        load_configs=False,
        connect_devices=False,
    )
    source_channel0 = SimpleNamespace(number=0, fnco_freq=125_000_000)
    source_channel1 = SimpleNamespace(number=0, fnco_freq=175_000_000)
    output_port0 = SimpleNamespace(
        id="B0.CTRL0",
        box_id="B0",
        number=1,
        type=PortType.CTRL,
        lo_freq=8_500_000_000,
        cnco_freq=1_617_187_500,
        channels=(source_channel0,),
        rfswitch="pass",
    )
    output_port1 = SimpleNamespace(
        id="B0.CTRL1",
        box_id="B0",
        number=2,
        type=PortType.CTRL,
        lo_freq=8_700_000_000,
        cnco_freq=1_700_000_000,
        channels=(source_channel1,),
        rfswitch="pass",
    )
    source_channel0.port = output_port0
    source_channel1.port = output_port1
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=9_000_000_000,
        cnco_freq=1_500_000_000,
        channels=(monitor_channel,),
        rfswitch="open",
    )
    monitor_channel.port = monitor_in_port
    box = SimpleNamespace(
        id="B0",
        ports=[output_port0, output_port1, monitor_in_port],
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
            **kwargs: Any,
        ) -> None:
            port = self._port_by_number[(box_id, port_number)]
            if "rfswitch" in kwargs and kwargs["rfswitch"] is not None:
                port.rfswitch = kwargs["rfswitch"]
            if "lo_freq" in kwargs:
                port.lo_freq = kwargs["lo_freq"]
            if "cnco_freq" in kwargs and kwargs["cnco_freq"] is not None:
                port.cnco_freq = kwargs["cnco_freq"]
            fnco_freqs = kwargs.get("fnco_freqs")
            if fnco_freqs is not None:
                for channel, fnco_freq in zip(port.channels, fnco_freqs, strict=True):
                    channel.fnco_freq = fnco_freq

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.port_config_calls: list[dict[str, Any]] = []
            self.runit_config_calls: list[dict[str, Any]] = []

        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

        def get_loopbacks_of_port(
            self,
            *,
            box_name: str,
            port_number: int,
        ) -> set[int]:
            _ = box_name
            if port_number == 4:
                return {1, 2}
            return set()

        def config_port(
            self,
            box_name: str,
            *,
            port: int,
            lo_freq_hz: int | None = None,
            cnco_freq_hz: int | None = None,
            cnco_locked_with: int | None = None,
            rfswitch: str | None = None,
        ) -> None:
            if lo_freq_hz is not None or cnco_locked_with is not None:
                self.port_config_calls.append(
                    {
                        "box_name": box_name,
                        "port": port,
                        "lo_freq_hz": lo_freq_hz,
                        "cnco_freq_hz": cnco_freq_hz,
                        "cnco_locked_with": cnco_locked_with,
                    }
                )
            if rfswitch is not None:
                control_system.set_port_params(
                    box_name,
                    port,
                    rfswitch=rfswitch,
                )

        def config_runit(
            self,
            box_name: str,
            *,
            port: int,
            runit: int,
            fnco_freq_hz: int | None = None,
        ) -> None:
            self.runit_config_calls.append(
                {
                    "box_name": box_name,
                    "port": port,
                    "runit": runit,
                    "fnco_freq_hz": fnco_freq_hz,
                }
            )

    control_system = _ControlSystemStub()
    backend_controller = _BackendControllerStub()
    experiment_system = SimpleNamespace(
        control_system=control_system,
        targets=[
            SimpleNamespace(label="Q00", channel=source_channel0),
            SimpleNamespace(label="Q01", channel=source_channel1),
        ],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
    )
    _bind_runtime(
        measurement,
        backend_controller=backend_controller,
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    built_source_labels: list[list[str]] = []
    run_source_labels: list[list[str]] = []

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, kwargs)
        built_source_labels.append(list(pulse_schedule.labels))
        return MeasurementSchedule(
            pulse_schedule=pulse_schedule.copy(),
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, quel1_options)
        labels = list(schedule.pulse_schedule.labels)
        run_source_labels.append(labels)
        if labels == ["Q00"]:
            assert monitor_in_port.lo_freq == output_port0.lo_freq
            assert monitor_in_port.cnco_freq == output_port0.cnco_freq
            assert monitor_channel.fnco_freq == source_channel0.fnco_freq
            payload_value = 1.0
        elif labels == ["Q01"]:
            assert monitor_in_port.lo_freq == output_port1.lo_freq
            assert monitor_in_port.cnco_freq == output_port1.cnco_freq
            assert monitor_channel.fnco_freq == source_channel1.fnco_freq
            payload_value = 2.0
        else:
            raise AssertionError(labels)
        return _make_measurement_result(
            data={
                "B0.MNTR0.IN": [
                    np.full((config.n_shots, 2), payload_value + 0.0j),
                ],
            },
            measurement_config=config,
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

    result = measurement.capture_loopback(
        schedule=PulseSchedule(["Q00", "Q01"]),
        n_shots=16,
        demodulation=False,
    )

    assert built_source_labels == [["Q00"], ["Q01"]]
    assert run_source_labels == [["Q00"], ["Q01"]]
    assert list(result.data) == ["Q00", "Q01"]
    assert result.data["Q00"][0].target == "Q00"
    assert result.data["Q00"][0].data.tolist() == [
        (1.0 + 0.0j),
        (1.0 + 0.0j),
    ]
    assert result.data["Q01"][0].target == "Q01"
    assert result.data["Q01"][0].data.tolist() == [
        (2.0 + 0.0j),
        (2.0 + 0.0j),
    ]
    assert backend_controller.port_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": 8_500_000_000,
            "cnco_freq_hz": None,
            "cnco_locked_with": 1,
        },
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": 8_700_000_000,
            "cnco_freq_hz": None,
            "cnco_locked_with": 2,
        },
    ]
    assert backend_controller.runit_config_calls == [
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 125_000_000,
        },
        {
            "box_name": "B0",
            "port": 4,
            "runit": 0,
            "fnco_freq_hz": 175_000_000,
        },
    ]


def test_capture_loopback_orders_merged_results_by_schedule_labels() -> None:
    """Given split loopback results, final keys should follow schedule labels."""
    config = _make_config(mode="avg", shots=1)
    result = _make_measurement_result(
        data={
            "Q08": [np.array([1.0 + 0.0j])],
            "Q09": [np.array([2.0 + 0.0j])],
            "RQ08": [np.array([3.0 + 0.0j])],
            "Q10": [np.array([4.0 + 0.0j])],
            "Q11": [np.array([5.0 + 0.0j])],
            "B0.MNTR0.IN": [np.array([6.0 + 0.0j])],
        },
        measurement_config=config,
        sampling_period=2.0,
    )

    ordered = MeasurementMonitorService._order_loopback_result_by_targets(  # noqa: SLF001
        result,
        target_order=["Q08", "Q09", "Q10", "Q11", "RQ08"],
    )

    assert list(ordered.data) == [
        "Q08",
        "Q09",
        "Q10",
        "Q11",
        "RQ08",
        "B0.MNTR0.IN",
    ]


def test_loopback_demodulation_preserves_awg_envelope() -> None:
    """Given a modulated pulse, software demodulation preserves its envelope."""
    sampling_period = 2.0
    frequency_ghz = 0.05
    sample_count = 512
    sample_times = np.arange(sample_count) * sampling_period
    envelope = np.zeros(sample_count, dtype=np.float64)
    envelope[128:384] = 1.0
    source = envelope * np.exp(1j * 2 * np.pi * frequency_ghz * sample_times)

    demodulated = MeasurementMonitorService._demodulate_loopback_capture(  # noqa: SLF001
        data=source,
        frequency_ghz=frequency_ghz,
        sampling_period=sampling_period,
    )

    np.testing.assert_allclose(np.abs(demodulated), envelope, atol=1e-12)
    assert np.count_nonzero(np.abs(demodulated) > 0.5) == 256


def test_monitor_demodulation_uses_residual_from_snapped_receiver_nco() -> None:
    """Given snapped monitor NCO, software demodulation uses the remaining offset."""
    source_channel = SimpleNamespace(number=0, fnco_freq=100_000_000)
    source_port = SimpleNamespace(
        id="B0.READ0.OUT",
        box_id="B0",
        number=1,
        type=PortType.READ_OUT,
        sideband="L",
        lo_freq=8_500_000_000,
        cnco_freq=2_203_125_000,
        channels=(source_channel,),
    )
    source_channel.port = source_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    monitor_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=6_000_000_000,
        cnco_freq=8 * NCO_STEP_HZ,
        channels=(monitor_channel,),
    )
    monitor_channel.port = monitor_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="quel1se-riken8"),
        ports=[source_port, monitor_port],
    )

    class _ControlSystemStub:
        boxes: ClassVar[list[Any]] = [box]

        @staticmethod
        def get_port_by_id(port_id: str) -> Any:
            if port_id != "B0.MNTR0.IN":
                raise KeyError(port_id)
            return monitor_port

        @staticmethod
        def get_box(box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

    class _BackendControllerStub:
        @staticmethod
        def dump_port(*, box_name: str, port_number: int) -> dict[str, Any]:
            _ = (box_name, port_number)
            return {}

    service = MeasurementMonitorService(
        context=SimpleNamespace(
            experiment_system=SimpleNamespace(
                control_system=_ControlSystemStub(),
                targets=[SimpleNamespace(label="RQ00", channel=source_channel)],
            ),
        ),
        session_service=SimpleNamespace(backend_controller=_BackendControllerStub()),
        execution_service=SimpleNamespace(),
    )
    source_setting = _LoopbackMonitorSourceSetting(
        label="RQ00",
        port=source_port,
        channel_number=0,
        lo_freq_hz=source_port.lo_freq,
        cnco_freq_hz=source_port.cnco_freq,
        fnco_freq_hz=source_channel.fnco_freq,
    )

    frequency = service._resolve_loopback_monitor_demodulation_frequency_ghz(  # noqa: SLF001
        capture_target="B0.MNTR0.IN",
        monitor_source_label="RQ00",
        monitor_source_setting=source_setting,
        pulse_schedule=PulseSchedule(["RQ00"]),
    )

    assert frequency == pytest.approx(0.009375)


def test_monitor_demodulation_uses_lsb_residual_for_direct_rf_source() -> None:
    """Given LSB monitor receiver, software residual uses the capture-side sign."""
    source_channel = SimpleNamespace(number=0, fnco_freq=-656_250_000)
    source_port = SimpleNamespace(
        id="B0.CTRL0",
        box_id="B0",
        number=1,
        type=PortType.CTRL,
        sideband=None,
        lo_freq=None,
        cnco_freq=4_921_875_000,
        channels=(source_channel,),
    )
    source_channel.port = source_port
    monitor_channel = SimpleNamespace(number=0, fnco_freq=0)
    target_frequency_hz = 4_364_006_032
    monitor_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        lo_freq=6_000_000_000,
        cnco_freq=1_640_625_000,
        channels=(monitor_channel,),
    )
    monitor_channel.port = monitor_port
    box = SimpleNamespace(
        id="B0",
        type=SimpleNamespace(value="quel1se-riken8"),
        ports=[source_port, monitor_port],
    )

    class _ControlSystemStub:
        boxes: ClassVar[list[Any]] = [box]

        @staticmethod
        def get_port_by_id(port_id: str) -> Any:
            if port_id != "B0.MNTR0.IN":
                raise KeyError(port_id)
            return monitor_port

        @staticmethod
        def get_box(box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

    class _BackendControllerStub:
        @staticmethod
        def dump_port(*, box_name: str, port_number: int) -> dict[str, Any]:
            _ = (box_name, port_number)
            return {}

    target = SimpleNamespace(
        label="Q00",
        frequency=target_frequency_hz * 1e-9,
        channel=source_channel,
    )
    service = MeasurementMonitorService(
        context=SimpleNamespace(
            experiment_system=SimpleNamespace(
                control_system=_ControlSystemStub(),
                targets=[target],
                get_target=lambda label: (
                    target
                    if label == target.label
                    else (_ for _ in ()).throw(KeyError(label))
                ),
            ),
        ),
        session_service=SimpleNamespace(backend_controller=_BackendControllerStub()),
        execution_service=SimpleNamespace(),
    )
    source_setting = _LoopbackMonitorSourceSetting(
        label="Q00",
        port=source_port,
        channel_number=0,
        lo_freq_hz=source_port.lo_freq,
        cnco_freq_hz=source_port.cnco_freq,
        fnco_freq_hz=source_channel.fnco_freq,
    )
    schedule = PulseSchedule(["Q00"])

    observed_hz = service._resolve_loopback_observed_frequency_hz(  # noqa: SLF001
        source_setting=source_setting,
        pulse_schedule=schedule,
    )
    frequency = service._resolve_loopback_monitor_demodulation_frequency_ghz(  # noqa: SLF001
        capture_target="B0.MNTR0.IN",
        monitor_source_label="Q00",
        monitor_source_setting=source_setting,
        pulse_schedule=schedule,
    )

    assert observed_hz == target_frequency_hz
    assert frequency == pytest.approx(-0.004631032)


def test_stability_service_computes_trimmed_monitor_statistics() -> None:
    """Given monitor captures, when statistics are computed, then trimmed data is used."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    result = _make_measurement_result(
        data={
            "B0.MNTR0.IN": [
                np.array(
                    [
                        [0.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 4.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
                    ]
                )
            ],
            "B0.READ0.IN": [
                np.array(
                    [
                        [10.0 + 0.0j, 10.0 + 0.0j],
                        [10.0 + 0.0j, 10.0 + 0.0j],
                    ]
                )
            ],
        },
        measurement_config=_make_config(mode="single"),
        sampling_period=2.0,
    )

    stats = measurement.stability_service.compute_monitor_statistics(
        result,
        targets=["B0.MNTR0.IN"],
        trim_samples=1,
    )

    assert len(stats) == 1
    assert stats[0].reference_target == "B0.MNTR0.IN"
    assert stats[0].covered_targets == ("B0.MNTR0.IN",)
    assert stats[0].monitor_target == "B0.MNTR0.IN"
    assert stats[0].amplitude_mean == pytest.approx(3.0)
    assert stats[0].n_samples == 4


def test_stability_service_keeps_monitor_phase_wrapped() -> None:
    """Given phase near the wrap boundary, when statistics are computed, then phase stays bounded."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    phases = np.array([np.pi - 0.1, -np.pi + 0.1])
    result = _make_measurement_result(
        data={"B0.MNTR0.IN": [np.exp(1j * phases)]},
        measurement_config=_make_config(mode="avg"),
        sampling_period=2.0,
    )

    stats = measurement.stability_service.compute_monitor_statistics(result)

    assert -np.pi <= stats[0].phase_mean_rad <= np.pi
    assert abs(abs(stats[0].phase_mean_rad) - np.pi) == pytest.approx(0.0)
    assert stats[0].phase_std_rad < 0.2
    assert stats[0].phase_resultant_length > 0.99


def test_stability_service_marks_ambiguous_monitor_phase() -> None:
    """Given opposite phases, when statistics are computed, then phase mean is undefined."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    result = _make_measurement_result(
        data={"B0.MNTR0.IN": [np.array([1.0 + 0.0j, -1.0 + 0.0j])]},
        measurement_config=_make_config(mode="avg"),
        sampling_period=2.0,
    )

    stats = measurement.stability_service.compute_monitor_statistics(result)

    assert stats[0].phase_resultant_length == pytest.approx(0.0)
    assert np.isnan(stats[0].phase_mean_rad)
    assert np.isnan(stats[0].phase_std_rad)


def test_loopback_capture_target_resolution_defaults_to_monitor_for_readout_outputs() -> (
    None
):
    """Given a readout output schedule, default loopback targets should stay monitor-only."""
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
    )
    read_in_port = SimpleNamespace(
        id="B0.READ0.IN",
        box_id="B0",
        number=2,
        type=PortType.READ_IN,
    )
    monitor_in_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port, monitor_in_port])

    def _resolve_qubit_label(label: str) -> str:
        if label == "RQ00":
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=SimpleNamespace(boxes=[box]),
        targets=[
            SimpleNamespace(
                label="RQ00",
                channel=SimpleNamespace(port=read_out_port),
            )
        ],
        read_in_targets=[
            SimpleNamespace(
                label="RQ00",
                channel=SimpleNamespace(port=read_in_port),
            )
        ],
        resolve_qubit_label=_resolve_qubit_label,
    )
    _bind_runtime(
        measurement,
        backend_controller=SimpleNamespace(),
        experiment_system=experiment_system,
    )

    service = measurement.monitor_service
    default_targets = service._resolve_loopback_capture_targets(  # noqa: SLF001
        schedule=PulseSchedule(["RQ00"]),
        include_read_in=False,
    )
    read_in_targets = service._resolve_loopback_capture_targets(  # noqa: SLF001
        schedule=PulseSchedule(["RQ00"]),
        include_read_in=True,
    )

    assert default_targets == ["B0.MNTR0.IN"]
    assert read_in_targets == ["B0.READ0.IN", "B0.MNTR0.IN"]


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
        channel=SimpleNamespace(port=SimpleNamespace(box_id="B0", type=PortType.CTRL)),
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
    captured_build_targets: list[list[str]] = []
    run_index = 0

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, pulse_schedule)
        captured_build_targets.append(list(cast(list[str], kwargs["capture_targets"])))
        return built_schedule

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        nonlocal run_index
        _ = (self, schedule, config, quel1_options)
        capture_targets = captured_build_targets[run_index]
        run_index += 1
        if capture_targets == ["B0.MNTR0.IN"]:
            assert read_in_port.rfswitch == "open"
            assert read_out_port.rfswitch == "pass"
            assert monitor_in_port.rfswitch == "loop"
        else:
            raise AssertionError(capture_targets)
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
    assert captured_build_targets == [["B0.MNTR0.IN"]]
    assert len(backend_controller.calls) == 2
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
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
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
        _ = measurement.capture_loopback(
            schedule=PulseSchedule(["RQ00"]),
            include_read_in=True,
        )

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
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
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

    result = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        include_read_in=True,
    )

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
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
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

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        n_shots=16,
        include_read_in=True,
    )

    assert backend_controller.init_calls == [["B0"]]


def test_capture_loopback_retries_with_monitor_only_after_read_in_e7_error() -> None:
    """Given READ_IN broken data errors, capture_loopback falls back to MNTR_IN only."""
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
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
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

    result = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        include_read_in=True,
    )

    assert "Q00" in result.data
    assert call_count["run"] == 3
    assert capture_target_calls[0] == ["B0.READ0.IN"]
    assert capture_target_calls[1] == ["B0.READ0.IN"]
    assert capture_target_calls[2] == ["B0.MNTR0.IN"]


def test_capture_loopback_uses_backend_dsp_demodulation_when_read_in_included() -> None:
    """Given included READ_IN capture, when running, then QuEL-1 DSP demodulation is enabled."""
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
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
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

    _ = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        n_shots=16,
        include_read_in=True,
    )

    options = cast(Quel1MeasurementOptions | None, called["quel1_options"])
    config = cast(MeasurementConfig, called["config"])
    assert config.shot_averaging is False
    assert config.n_shots == 16
    assert options is not None
    assert options.demodulation is True


def test_capture_loopback_uses_backend_demodulated_read_in_before_shot_averaging() -> (
    None
):
    """Given read-in waveforms, capture_loopback should use backend demodulation before averaging."""
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
        channels=(SimpleNamespace(fnco_freq=0),),
    )
    monitor_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        rfswitch="open",
        channels=(SimpleNamespace(fnco_freq=0),),
    )
    box = SimpleNamespace(id="B0", ports=[read_out_port, read_in_port, monitor_port])

    class _ControlSystemStub:
        boxes: ClassVar[list[Any]] = [box]

        @staticmethod
        def get_port_by_id(port_id: str) -> Any:
            ports = {
                read_out_port.id: read_out_port,
                read_in_port.id: read_in_port,
                monitor_port.id: monitor_port,
            }
            try:
                return ports[port_id]
            except KeyError:
                raise KeyError(port_id) from None

        @staticmethod
        def get_box(box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

    class _BackendControllerStub:
        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

    schedule_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_out_port),
    )
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=read_in_port),
    )

    def _resolve_qubit_label(label: str) -> str:
        if label in {"Q00", "RQ00"}:
            return "Q00"
        raise ValueError(label)

    experiment_system = SimpleNamespace(
        control_system=_ControlSystemStub(),
        targets=[schedule_target],
        read_in_targets=[read_in_target],
        resolve_qubit_label=_resolve_qubit_label,
        get_nco_frequency=lambda _label: 0.0,
        get_target=lambda _label: SimpleNamespace(sideband="U"),
    )
    _bind_runtime(
        measurement,
        backend_controller=_BackendControllerStub(),
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    frequency = 0.25
    sampling_period = 1.0
    samples = np.arange(4, dtype=np.float64)
    rotating = np.exp(1j * 2 * np.pi * frequency * sampling_period * samples)
    raw_waveforms = np.vstack([rotating, rotating])

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, kwargs)
        schedule = pulse_schedule.copy()
        schedule.set_frequency("RQ00", frequency)
        return MeasurementSchedule(
            pulse_schedule=schedule,
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule)
        options = cast(Quel1MeasurementOptions, quel1_options)
        returned_waveforms = (
            np.ones_like(raw_waveforms) if options.demodulation else raw_waveforms
        )
        return _make_measurement_result(
            data={"Q00": [returned_waveforms]},
            measurement_config=config,
            sampling_period=sampling_period,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    result = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        n_shots=2,
        include_read_in=True,
    )

    capture = result.data["Q00"][0]
    assert result.measurement_config.shot_averaging is True
    assert capture.config.shot_averaging is True
    assert capture.data.shape == (4,)
    np.testing.assert_allclose(capture.data, np.ones(4, dtype=np.complex128))
    assert abs(np.mean(capture.data)) == pytest.approx(1.0)

    raw_result = measurement.capture_loopback(
        schedule=PulseSchedule(["RQ00"]),
        n_shots=2,
        shot_averaging=False,
        demodulation=False,
        include_read_in=True,
    )
    raw_capture = raw_result.data["Q00"][0]
    assert raw_result.measurement_config.shot_averaging is False
    assert raw_capture.config.shot_averaging is False
    assert raw_capture.data.shape == (2, 4)
    np.testing.assert_allclose(raw_capture.data, raw_waveforms)


def test_capture_loopback_does_not_source_demodulate_monitor_capture() -> None:
    """Given monitor DC capture, source frequency metadata should not create a sine wave."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    monitor_port = SimpleNamespace(
        id="B0.MNTR0.IN",
        box_id="B0",
        number=4,
        type=PortType.MNTR_IN,
        rfswitch="open",
        channels=(SimpleNamespace(fnco_freq=0),),
    )
    box = SimpleNamespace(id="B0", ports=[monitor_port])

    class _ControlSystemStub:
        boxes: ClassVar[list[Any]] = [box]

        @staticmethod
        def get_port_by_id(port_id: str) -> Any:
            if port_id != "B0.MNTR0.IN":
                raise KeyError(port_id)
            return monitor_port

        @staticmethod
        def get_box(box_id: str) -> Any:
            if box_id != "B0":
                raise KeyError(box_id)
            return box

    class _BackendControllerStub:
        def initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
            _ = box_ids

    source_channel = SimpleNamespace(number=0, fnco_freq=0)
    source_port = SimpleNamespace(
        id="B0.CTRL0.OUT",
        box_id="B0",
        number=1,
        type=PortType.CTRL,
        channels=(source_channel,),
    )
    source_channel.port = source_port
    schedule_target = SimpleNamespace(
        label="Q00",
        channel=source_channel,
    )
    experiment_system = SimpleNamespace(
        control_system=_ControlSystemStub(),
        targets=[schedule_target],
        read_in_targets=[],
        resolve_qubit_label=lambda label: label,
        get_awg_frequency=lambda _label: 0.25,
    )
    _bind_runtime(
        measurement,
        backend_controller=_BackendControllerStub(),
        experiment_system=experiment_system,
    )

    execution_service = measurement.execution_service
    raw_waveforms = np.ones((2, 4), dtype=np.complex128)

    def _build(
        self: MeasurementExecutionService,
        pulse_schedule: PulseSchedule,
        **kwargs: Any,
    ) -> MeasurementSchedule:
        _ = (self, kwargs)
        schedule = pulse_schedule.copy()
        schedule.set_frequency("Q00", 0.25)
        return MeasurementSchedule(
            pulse_schedule=schedule,
            capture_schedule=CaptureSchedule(captures=[]),
        )

    async def _run(
        self: MeasurementExecutionService,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        _ = (self, schedule, quel1_options)
        return _make_measurement_result(
            data={"B0.MNTR0.IN": [raw_waveforms]},
            measurement_config=config,
            sampling_period=1.0,
        )

    execution_service.build_measurement_schedule = MethodType(
        _build,
        execution_service,
    )
    execution_service.run_measurement = MethodType(
        _run,
        execution_service,
    )

    result = measurement.capture_loopback(
        schedule=PulseSchedule(["Q00"]),
        n_shots=2,
        demodulation=False,
    )

    capture = result.data["Q00"][0]
    assert capture.config.shot_averaging is True
    assert capture.data.shape == (4,)
    assert capture.target == "Q00"
    np.testing.assert_allclose(capture.data, np.ones(4, dtype=np.complex128))


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
