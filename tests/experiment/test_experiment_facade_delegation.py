"""Tests for Experiment facade delegation methods."""

from __future__ import annotations

from typing import Any

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


class _ExperimentContextStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def print_environment(self, verbose: bool | None = None) -> None:
        self.calls.append(("print_environment", {"verbose": verbose}))

    def print_boxes(self) -> None:
        self.calls.append(("print_boxes", {}))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", {}))

    def register_custom_target(self, **kwargs: Any) -> None:
        self.calls.append(("register_custom_target", kwargs))


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


def test_disconnect_delegates_to_context() -> None:
    """Given disconnect call, when called, then it delegates to experiment context."""
    exp = object.__new__(Experiment)
    context_stub = _ExperimentContextStub()
    exp.__dict__["_experiment_context"] = context_stub

    exp.disconnect()

    assert context_stub.calls == [("disconnect", {})]


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
