"""Tests for Experiment facade delegation methods."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from qubex.experiment.experiment import Experiment


class _CalibrationServiceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def calibrate_1q(self, **kwargs: Any) -> str:
        self.calls.append(("calibrate_1q", kwargs))
        return "calibrate_1q_result"

    def calibrate_2q(self, **kwargs: Any) -> str:
        self.calls.append(("calibrate_2q", kwargs))
        return "calibrate_2q_result"


class _BenchmarkingServiceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def benchmark_1q(self, **kwargs: Any) -> None:
        self.calls.append(("benchmark_1q", kwargs))

    def benchmark_2q(self, **kwargs: Any) -> None:
        self.calls.append(("benchmark_2q", kwargs))


class _MeasurementServiceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def measure_idle_states(self, **kwargs: Any) -> str:
        self.calls.append(("measure_idle_states", kwargs))
        return "measure_idle_states_result"

    def execute(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("execute", {"schedule": schedule, **kwargs}))
        return "execute_result"

    def measure(self, sequence: object, **kwargs: Any) -> str:
        self.calls.append(("measure", {"sequence": sequence, **kwargs}))
        return "measure_result"

    def capture_loopback(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("capture_loopback", {"schedule": schedule, **kwargs}))
        return "capture_loopback_result"

    def build_measurement_schedule(self, pulse_schedule: object, **kwargs: Any) -> str:
        self.calls.append(
            ("build_measurement_schedule", {"pulse_schedule": pulse_schedule, **kwargs})
        )
        return "build_measurement_schedule_result"

    async def run_measurement(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("run_measurement", {"schedule": schedule, **kwargs}))
        return "run_measurement_result"

    async def run_sweep_measurement(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("run_sweep_measurement", {"schedule": schedule, **kwargs}))
        return "run_sweep_measurement_result"

    async def run_ndsweep_measurement(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("run_ndsweep_measurement", {"schedule": schedule, **kwargs}))
        return "run_ndsweep_measurement_result"

    def sweep_parameter(self, **kwargs: Any) -> str:
        self.calls.append(("sweep_parameter", kwargs))
        return "sweep_parameter_result"

    def state_tomography(self, **kwargs: Any) -> str:
        self.calls.append(("state_tomography", kwargs))
        return "state_tomography_result"


class _ExperimentContextStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def print_environment(self, verbose: bool | None = None) -> None:
        self.calls.append(("print_environment", {"verbose": verbose}))

    def print_boxes(self) -> None:
        self.calls.append(("print_boxes", {}))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", {}))

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        self.calls.append(
            ("connect", {"sync_clocks": sync_clocks, "parallel": parallel})
        )

    def reload(self) -> None:
        self.calls.append(("reload", {}))

    def register_custom_target(self, **kwargs: Any) -> None:
        self.calls.append(("register_custom_target", kwargs))


class _SessionServiceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def disconnect(self) -> None:
        self.calls.append(("disconnect", {}))

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        self.calls.append(
            ("connect", {"sync_clocks": sync_clocks, "parallel": parallel})
        )

    def reload(self) -> None:
        self.calls.append(("reload", {}))

    def configure(self, **kwargs: Any) -> None:
        self.calls.append(("configure", kwargs))

    def linkup(self, **kwargs: Any) -> None:
        self.calls.append(("linkup", kwargs))

    def resync_clocks(self, **kwargs: Any) -> None:
        self.calls.append(("resync_clocks", kwargs))

    def reset_awg_and_capunits(self, **kwargs: Any) -> None:
        self.calls.append(("reset_awg_and_capunits", kwargs))


class _BenchmarkingServicePropertyStub:
    def __init__(self) -> None:
        self.clifford_generator = object()
        self.clifford: dict[str, Any] = {"I": object()}


def test_calibrate_1q_delegates_to_calibration_service() -> None:
    """Given calibrate_1q arguments, when called, then it delegates to calibration service."""
    exp = object.__new__(Experiment)
    calibration_stub = _CalibrationServiceStub()
    exp.__dict__["_calibration_service"] = calibration_stub

    result = exp.calibrate_1q(
        targets=["Q00", "Q01"],
        n_shots=1024,
        shot_interval=120.0,
        plot=False,
        coarse=True,
    )

    assert result == "calibrate_1q_result"
    assert calibration_stub.calls == [
        (
            "calibrate_1q",
            {
                "targets": ["Q00", "Q01"],
                "shots": 1024,
                "interval": 120.0,
                "plot": False,
                "coarse": True,
            },
        )
    ]


def test_calibrate_2q_delegates_to_calibration_service() -> None:
    """Given calibrate_2q arguments, when called, then it delegates to calibration service."""
    exp = object.__new__(Experiment)
    calibration_stub = _CalibrationServiceStub()
    exp.__dict__["_calibration_service"] = calibration_stub

    result = exp.calibrate_2q(
        targets=["Q00-Q01"],
        cr_calib_params={"Q00-Q01": {"n_iterations": 3}},
        n_shots=2048,
        shot_interval=150.0,
        plot=True,
    )

    assert result == "calibrate_2q_result"
    assert calibration_stub.calls == [
        (
            "calibrate_2q",
            {
                "targets": ["Q00-Q01"],
                "cr_calib_params": {"Q00-Q01": {"n_iterations": 3}},
                "shots": 2048,
                "interval": 150.0,
                "plot": True,
            },
        )
    ]


def test_benchmark_1q_delegates_to_benchmarking_service() -> None:
    """Given benchmark_1q arguments, when called, then it delegates to benchmarking service."""
    exp = object.__new__(Experiment)
    benchmarking_stub = _BenchmarkingServiceStub()
    exp.__dict__["_benchmarking_service"] = benchmarking_stub

    result = exp.benchmark_1q(
        targets=["Q00", "Q01"],
        n_trials=10,
        in_parallel=True,
        n_shots=512,
        shot_interval=200.0,
        plot=False,
        save_image=True,
    )

    assert result is None
    assert benchmarking_stub.calls == [
        (
            "benchmark_1q",
            {
                "targets": ["Q00", "Q01"],
                "n_trials": 10,
                "in_parallel": True,
                "shots": 512,
                "interval": 200.0,
                "plot": False,
                "save_image": True,
            },
        )
    ]


def test_benchmark_2q_delegates_to_benchmarking_service() -> None:
    """Given benchmark_2q arguments, when called, then it delegates to benchmarking service."""
    exp = object.__new__(Experiment)
    benchmarking_stub = _BenchmarkingServiceStub()
    exp.__dict__["_benchmarking_service"] = benchmarking_stub

    result = exp.benchmark_2q(
        targets=["Q00-Q01"],
        n_trials=12,
        in_parallel=False,
        n_shots=1024,
        shot_interval=240.0,
        plot=True,
        save_image=False,
    )

    assert result is None
    assert benchmarking_stub.calls == [
        (
            "benchmark_2q",
            {
                "targets": ["Q00-Q01"],
                "n_trials": 12,
                "in_parallel": False,
                "shots": 1024,
                "interval": 240.0,
                "plot": True,
                "save_image": False,
            },
        )
    ]


def test_print_environment_delegates_to_context() -> None:
    """Given print_environment args, when called, then it delegates to experiment context."""
    exp = object.__new__(Experiment)
    context_stub = _ExperimentContextStub()
    exp.__dict__["_experiment_context"] = context_stub

    exp.print_environment(verbose=False)

    assert context_stub.calls == [
        (
            "print_environment",
            {
                "verbose": False,
            },
        )
    ]


def test_print_boxes_delegates_to_context() -> None:
    """Given print_boxes call, when called, then it delegates to experiment context."""
    exp = object.__new__(Experiment)
    context_stub = _ExperimentContextStub()
    exp.__dict__["_experiment_context"] = context_stub

    exp.print_boxes()

    assert context_stub.calls == [("print_boxes", {})]


def test_run_executes_task_with_experiment_instance() -> None:
    """Given experiment task, when run is called, then task executes with the experiment instance."""
    exp = object.__new__(Experiment)
    called: dict[str, object] = {}

    class _Task:
        def execute(self, exp: Experiment) -> str:
            called["experiment"] = exp
            return "task_result"

    result = exp.run(_Task())

    assert result == "task_result"
    assert called["experiment"] is exp


def test_clifford_generator_property_delegates_to_benchmarking_service() -> None:
    """Given benchmarking service, when clifford_generator is accessed, then it returns delegated value."""
    exp = object.__new__(Experiment)
    benchmarking_stub = _BenchmarkingServicePropertyStub()
    exp.__dict__["_benchmarking_service"] = benchmarking_stub

    assert exp.clifford_generator is benchmarking_stub.clifford_generator


def test_clifford_property_delegates_to_benchmarking_service() -> None:
    """Given benchmarking service, when clifford is accessed, then it returns delegated value."""
    exp = object.__new__(Experiment)
    benchmarking_stub = _BenchmarkingServicePropertyStub()
    exp.__dict__["_benchmarking_service"] = benchmarking_stub

    assert exp.clifford is benchmarking_stub.clifford


def test_disconnect_delegates_to_session_service() -> None:
    """Given disconnect call, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.disconnect()

    assert session_stub.calls == [("disconnect", {})]


def test_connect_delegates_to_session_service() -> None:
    """Given connect args, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.connect(sync_clocks=False, parallel=True)

    assert session_stub.calls == [("connect", {"sync_clocks": False, "parallel": True})]


def test_reload_delegates_to_session_service() -> None:
    """Given reload call, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.reload()

    assert session_stub.calls == [("reload", {})]


def test_capture_loopback_delegates_to_measurement_service() -> None:
    """Given loopback capture arguments, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    schedule = cast(Any, object())

    result = exp.capture_loopback(schedule, n_shots=128)

    assert result == "capture_loopback_result"
    assert measurement_stub.calls == [
        (
            "capture_loopback",
            {
                "schedule": schedule,
                "n_shots": 128,
            },
        )
    ]


def test_execute_delegates_new_shot_arguments_to_measurement_service() -> None:
    """Given n_shots args, when execute is called, then it delegates canonical shot keys."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    schedule = cast(Any, object())

    _ = exp.execute(schedule=schedule, n_shots=256, shot_interval=120.0)

    assert measurement_stub.calls == [
        (
            "execute",
            {
                "schedule": schedule,
                "frequencies": None,
                "mode": None,
                "n_shots": 256,
                "shot_interval": 120.0,
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramptime": None,
                "readout_drag_coeff": None,
                "readout_ramp_type": None,
                "add_last_measurement": None,
                "add_pump_pulses": None,
                "enable_dsp_demodulation": None,
                "enable_dsp_sum": None,
                "enable_dsp_classification": None,
                "line_param0": None,
                "line_param1": None,
                "reset_awg_and_capunits": None,
                "plot": None,
            },
        )
    ]


def test_measure_delegates_legacy_shot_arguments_to_measurement_service() -> None:
    """Given legacy shot kwargs, when measure is called, then deprecated keys are delegated."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sequence = cast(Any, {"Q00": [0.0 + 0.0j]})

    _ = exp.measure(sequence=sequence, shots=64, interval=256.0)

    assert measurement_stub.calls == [
        (
            "measure",
            {
                "sequence": sequence,
                "frequencies": None,
                "initial_states": None,
                "mode": None,
                "n_shots": None,
                "shot_interval": None,
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramptime": None,
                "readout_drag_coeff": None,
                "readout_ramp_type": None,
                "add_pump_pulses": None,
                "enable_dsp_demodulation": None,
                "enable_dsp_sum": None,
                "enable_dsp_classification": None,
                "line_param0": None,
                "line_param1": None,
                "reset_awg_and_capunits": None,
                "plot": None,
                "shots": 64,
                "interval": 256.0,
            },
        )
    ]


def test_build_measurement_schedule_delegates_to_measurement_service() -> None:
    """Given schedule-build arguments, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    pulse_schedule = cast(Any, object())

    result = exp.build_measurement_schedule(
        pulse_schedule,
        frequencies={"Q00": 5.1},
        readout_amplification=True,
        final_measurement=False,
        capture_placement="pulse_aligned",
        plot=False,
    )

    assert result == "build_measurement_schedule_result"
    assert measurement_stub.calls == [
        (
            "build_measurement_schedule",
            {
                "pulse_schedule": pulse_schedule,
                "frequencies": {"Q00": 5.1},
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramp_time": None,
                "readout_ramp_type": None,
                "readout_drag_coeff": None,
                "readout_amplification": True,
                "final_measurement": False,
                "capture_placement": "pulse_aligned",
                "capture_targets": None,
                "plot": False,
            },
        )
    ]


def test_run_measurement_delegates_to_measurement_service() -> None:
    """Given async run_measurement args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    schedule = cast(Any, object())

    result = asyncio.run(
        exp.run_measurement(
            schedule,
            frequencies={"Q00": 5.2},
            final_measurement=True,
            n_shots=256,
        )
    )

    assert result == "run_measurement_result"
    assert measurement_stub.calls == [
        (
            "run_measurement",
            {
                "schedule": schedule,
                "frequencies": {"Q00": 5.2},
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramp_time": None,
                "readout_ramp_type": None,
                "readout_drag_coeff": None,
                "readout_amplification": None,
                "final_measurement": True,
                "n_shots": 256,
                "shot_interval": None,
                "shot_averaging": None,
                "time_integration": None,
                "state_classification": None,
            },
        )
    ]


def test_run_sweep_measurement_delegates_to_measurement_service() -> None:
    """Given async run_sweep args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sweep_values = [0.1, 0.2]
    schedule = cast(Any, object())

    result = asyncio.run(
        exp.run_sweep_measurement(
            schedule,
            sweep_values=sweep_values,
            final_measurement=True,
            shot_averaging=False,
            plot=True,
            enable_tqdm=True,
        )
    )

    assert result == "run_sweep_measurement_result"
    assert measurement_stub.calls == [
        (
            "run_sweep_measurement",
            {
                "schedule": schedule,
                "sweep_values": sweep_values,
                "frequencies": None,
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramp_time": None,
                "readout_ramp_type": None,
                "readout_drag_coeff": None,
                "readout_amplification": None,
                "final_measurement": True,
                "plot": True,
                "enable_tqdm": True,
                "n_shots": None,
                "shot_interval": None,
                "shot_averaging": False,
                "time_integration": None,
                "state_classification": None,
            },
        )
    ]


def test_run_ndsweep_measurement_delegates_to_measurement_service() -> None:
    """Given async run_ndsweep args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sweep_points = {"x": [0.1, 0.2], "y": [1, 2]}
    sweep_axes = ("x", "y")
    schedule = cast(Any, object())

    result = asyncio.run(
        exp.run_ndsweep_measurement(
            schedule,
            sweep_points=sweep_points,
            sweep_axes=sweep_axes,
            readout_amplification=True,
            state_classification=True,
        )
    )

    assert result == "run_ndsweep_measurement_result"
    assert measurement_stub.calls == [
        (
            "run_ndsweep_measurement",
            {
                "schedule": schedule,
                "sweep_points": sweep_points,
                "sweep_axes": sweep_axes,
                "frequencies": None,
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "readout_ramp_time": None,
                "readout_ramp_type": None,
                "readout_drag_coeff": None,
                "readout_amplification": True,
                "final_measurement": None,
                "plot": None,
                "enable_tqdm": None,
                "n_shots": None,
                "shot_interval": None,
                "shot_averaging": None,
                "time_integration": None,
                "state_classification": True,
            },
        )
    ]


def test_configure_delegates_to_session_service() -> None:
    """Given configure args, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.configure(box_ids="Q2A", exclude="Q00", mode="ge-cr-cr")

    assert session_stub.calls == [
        (
            "configure",
            {
                "box_ids": "Q2A",
                "exclude": "Q00",
                "mode": "ge-cr-cr",
            },
        )
    ]


def test_linkup_warns_deprecation_and_delegates_to_session_service() -> None:
    """Given linkup args, when called, then deprecation warning is emitted and session service is delegated."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    with pytest.warns(DeprecationWarning, match="measurement\\.linkup|connect\\(\\)"):
        exp.linkup(box_ids=["Q2A"], noise_threshold=100)

    assert session_stub.calls == [
        (
            "linkup",
            {
                "box_ids": ["Q2A"],
                "noise_threshold": 100,
            },
        )
    ]


def test_resync_clocks_delegates_to_session_service() -> None:
    """Given resync args, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.resync_clocks(box_ids=["Q2A"])

    assert session_stub.calls == [
        (
            "resync_clocks",
            {
                "box_ids": ["Q2A"],
            },
        )
    ]


def test_reset_awg_and_capunits_delegates_to_session_service() -> None:
    """Given reset args, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

    exp.reset_awg_and_capunits(box_ids="Q2A", qubits=["Q00"])

    assert session_stub.calls == [
        (
            "reset_awg_and_capunits",
            {
                "box_ids": "Q2A",
                "qubits": ["Q00"],
            },
        )
    ]


def test_measure_idle_states_delegates_to_measurement_service() -> None:
    """Given measure_idle_states args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub

    result = exp.measure_idle_states(
        targets=["Q00", "Q01"],
        shots=512,
        interval=200.0,
        readout_amplitudes={"Q00": 0.01, "Q01": 0.02},
        add_pump_pulses=False,
        plot=True,
    )

    assert result == "measure_idle_states_result"
    assert measurement_stub.calls == [
        (
            "measure_idle_states",
            {
                "targets": ["Q00", "Q01"],
                "n_shots": None,
                "shot_interval": None,
                "shots": 512,
                "interval": 200.0,
                "readout_amplitudes": {"Q00": 0.01, "Q01": 0.02},
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "add_pump_pulses": False,
                "plot": True,
            },
        )
    ]


def test_sweep_parameter_delegates_legacy_shot_arguments_to_measurement_service() -> (
    None
):
    """Given legacy shot kwargs, when sweep_parameter is called, then deprecated keys are delegated."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sequence = cast(Any, lambda value: {"Q00": [value + 0.0j]})

    _ = exp.sweep_parameter(
        sequence=sequence,
        sweep_range=[0.1, 0.2],
        shots=32,
        interval=44.0,
    )

    assert measurement_stub.calls == [
        (
            "sweep_parameter",
            {
                "sequence": sequence,
                "sweep_range": [0.1, 0.2],
                "repetitions": None,
                "frequencies": None,
                "initial_states": None,
                "rabi_level": None,
                "n_shots": None,
                "shot_interval": None,
                "readout_amplitudes": None,
                "readout_duration": None,
                "readout_pre_margin": None,
                "readout_post_margin": None,
                "plot": None,
                "enable_tqdm": None,
                "title": None,
                "xlabel": None,
                "ylabel": None,
                "xaxis_type": None,
                "yaxis_type": None,
                "shots": 32,
                "interval": 44.0,
            },
        )
    ]


def test_state_tomography_delegates_canonical_shot_arguments_to_measurement_service() -> (
    None
):
    """Given canonical shot kwargs, when state_tomography is called, then canonical keys are delegated."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sequence = cast(Any, {"Q00": [0.0 + 0.0j]})

    result = exp.state_tomography(
        sequence=sequence,
        n_shots=128,
        shot_interval=64.0,
        plot=False,
    )

    assert result == "state_tomography_result"
    assert measurement_stub.calls == [
        (
            "state_tomography",
            {
                "sequence": sequence,
                "x90": None,
                "initial_state": None,
                "n_shots": 128,
                "shot_interval": 64.0,
                "reset_awg_and_capunits": None,
                "method": None,
                "use_zvalues": None,
                "plot": False,
            },
        )
    ]


def test_register_custom_target_delegates_to_context() -> None:
    """Given custom-target args, when called, then it delegates to experiment context."""
    exp = object.__new__(Experiment)
    context_stub = _ExperimentContextStub()
    exp.__dict__["_experiment_context"] = context_stub

    exp.register_custom_target(
        label="CUSTOM",
        frequency=5.1,
        box_id="B0",
        port_number=2,
        channel_number=0,
        qubit_label="Q00",
    )

    assert context_stub.calls == [
        (
            "register_custom_target",
            {
                "label": "CUSTOM",
                "frequency": 5.1,
                "box_id": "B0",
                "port_number": 2,
                "channel_number": 0,
                "qubit_label": "Q00",
                "target_type": None,
                "update_lsi": None,
            },
        )
    ]
