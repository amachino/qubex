"""Tests for Experiment facade delegation methods."""

from __future__ import annotations

import asyncio
from typing import Any

from qxpulse import PulseSchedule

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

    async def execute_async(self, schedule: object, **kwargs: Any) -> str:
        self.calls.append(("execute_async", {"schedule": schedule, **kwargs}))
        return "execute_async_result"

    def measure(self, sequence: object, **kwargs: Any) -> str:
        self.calls.append(("measure", {"sequence": sequence, **kwargs}))
        return "measure_result"

    async def measure_async(self, sequence: object, **kwargs: Any) -> str:
        self.calls.append(("measure_async", {"sequence": sequence, **kwargs}))
        return "measure_async_result"


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
        shots=1024,
        interval=120.0,
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
        shots=2048,
        interval=150.0,
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
        shots=512,
        interval=200.0,
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
        shots=1024,
        interval=240.0,
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


def test_linkup_delegates_to_session_service() -> None:
    """Given linkup args, when called, then it delegates to session service."""
    exp = object.__new__(Experiment)
    session_stub = _SessionServiceStub()
    exp.__dict__["_session_service"] = session_stub

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


def test_execute_async_delegates_to_measurement_service() -> None:
    """Given execute_async args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    schedule = PulseSchedule(["Q00"])

    result = asyncio.run(
        exp.execute_async(
            schedule=schedule,
            shots=512,
            plot=False,
        )
    )

    assert result == "execute_async_result"
    assert measurement_stub.calls == [
        (
            "execute_async",
            {
                "schedule": schedule,
                "frequencies": None,
                "mode": None,
                "shots": 512,
                "interval": None,
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
                "plot": False,
            },
        )
    ]


def test_measure_async_delegates_to_measurement_service() -> None:
    """Given measure_async args, when called, then it delegates to measurement service."""
    exp = object.__new__(Experiment)
    measurement_stub = _MeasurementServiceStub()
    exp.__dict__["_measurement_service"] = measurement_stub
    sequence = {"Q00": [0.0 + 0.0j]}

    result = asyncio.run(
        exp.measure_async(
            sequence=sequence,
            shots=256,
            plot=True,
        )
    )

    assert result == "measure_async_result"
    assert measurement_stub.calls == [
        (
            "measure_async",
            {
                "sequence": sequence,
                "frequencies": None,
                "initial_states": None,
                "mode": None,
                "shots": 256,
                "interval": None,
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
                "plot": True,
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
