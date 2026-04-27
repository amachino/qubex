"""Tests for best-effort experiment workflow execution."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

from qubex.experiment.models.result import Result
from qubex.experiment.services.benchmarking_service import BenchmarkingService
from qubex.experiment.services.calibration_service import CalibrationService
from qubex.experiment.services.characterization_service import CharacterizationService


def test_calibrate_2q_continues_after_pair_resolution_failure() -> None:
    """Given one invalid CR target, when calibrating 2Q gates, then later targets still run."""
    service = cast(Any, object.__new__(CalibrationService))
    calls: list[tuple[str, str]] = []

    def _cr_pair(label: str) -> tuple[str, str]:
        if label == "BAD_CR":
            raise RuntimeError("unknown CR target")
        return ("Q01", "Q02")

    service.__dict__["_experiment_context"] = SimpleNamespace(
        cr_labels=["BAD_CR", "CR12"],
        cr_pair=_cr_pair,
        save_calib_note=lambda: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _obtain_cr_params(**kwargs: object) -> Result:
        calls.append(
            (
                cast(str, kwargs["control_qubit"]),
                cast(str, kwargs["target_qubit"]),
            )
        )
        return Result(data={"ok": True})

    service.__dict__["obtain_cr_params"] = _obtain_cr_params
    service.__dict__["calibrate_zx90"] = lambda **_kwargs: Result(data={"ok": True})

    result = service.calibrate_2q(targets=["BAD_CR", "CR12"], plot=False)

    assert calls == [("Q01", "Q02")]
    data = cast(dict[str, Any], result.data)
    assert "Q01-Q02" in data["obtain_cr_params"]


def test_characterize_1q_continues_after_failed_step() -> None:
    """Given one failed 1Q characterization step, when running the suite, then remaining steps still run."""
    service = cast(Any, object.__new__(CharacterizationService))
    calls: list[tuple[str, str]] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00", "Q01"]
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _t1_experiment(
        self: CharacterizationService,
        target: str,
        **_kwargs: object,
    ) -> Result:
        calls.append(("t1", target))
        if target == "Q00":
            raise RuntimeError("t1 failed")
        return Result(data={target: SimpleNamespace(t1=1000.0)})

    def _t2_experiment(
        self: CharacterizationService,
        target: str,
        **_kwargs: object,
    ) -> Result:
        calls.append(("t2", target))
        return Result(data={target: SimpleNamespace(t2=2000.0)})

    def _ramsey_experiment(
        self: CharacterizationService,
        target: str,
        **_kwargs: object,
    ) -> Result:
        calls.append(("ramsey", target))
        return Result(data={target: SimpleNamespace(t2=3000.0, bare_freq=5.0)})

    service.__dict__["t1_experiment"] = MethodType(_t1_experiment, service)
    service.__dict__["t2_experiment"] = MethodType(_t2_experiment, service)
    service.__dict__["ramsey_experiment"] = MethodType(_ramsey_experiment, service)

    result = service.characterize_1q(targets=["Q00", "Q01"], plot=False)

    assert calls == [
        ("t1", "Q00"),
        ("t2", "Q00"),
        ("ramsey", "Q00"),
        ("t1", "Q01"),
        ("t2", "Q01"),
        ("ramsey", "Q01"),
    ]
    data = cast(dict[str, Any], result.data)
    assert "Q00" not in data["t1_experiment"]
    assert "Q00" in data["t2_experiment"]
    assert "Q01" in data["t1_experiment"]


def test_characterize_2q_continues_after_failed_target() -> None:
    """Given one failed 2Q characterization target, when running the suite, then later targets still run."""
    service = cast(Any, object.__new__(CharacterizationService))
    calls: list[str] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        get_edge_labels=lambda *, in_same_mux: ["Q00-Q01", "Q02-Q03"],
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _obtain_coupling_strength(
        self: CharacterizationService,
        control_qubit: str,
        target_qubit: str,
        **_kwargs: object,
    ) -> Result:
        target = f"{control_qubit}-{target_qubit}"
        calls.append(target)
        if target == "Q00-Q01":
            raise RuntimeError("coupling failed")
        return Result(data={"g": 1.0e-3, "xi": 2.0e-6})

    service.__dict__["obtain_coupling_strength"] = MethodType(
        _obtain_coupling_strength,
        service,
    )

    result = service.characterize_2q(targets=["Q00-Q01", "Q02-Q03"], plot=False)

    assert calls == ["Q00-Q01", "Q02-Q03"]
    data = cast(dict[str, Any], result.data)
    assert "Q00-Q01" not in data["obtain_coupling_strength"]
    assert data["obtain_coupling_strength"]["Q02-Q03"] == {
        "g": 1.0e-3,
        "xi": 2.0e-6,
    }


def test_benchmark_1q_continues_after_failed_clifford() -> None:
    """Given one failed 1Q benchmark step, when running the suite, then remaining steps still run."""
    service = cast(Any, object.__new__(BenchmarkingService))
    calls: list[tuple[str, str]] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00", "Q01"]
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _interleaved_randomized_benchmarking(
        self: BenchmarkingService,
        targets: str,
        *,
        interleaved_clifford: str,
        **_kwargs: object,
    ) -> Result:
        calls.append((targets, interleaved_clifford))
        if targets == "Q00" and interleaved_clifford == "X90":
            raise RuntimeError("benchmark failed")
        return Result(data={targets: {"ok": True}})

    service.__dict__["interleaved_randomized_benchmarking"] = MethodType(
        _interleaved_randomized_benchmarking,
        service,
    )

    service.benchmark_1q(targets=["Q00", "Q01"], in_parallel=False, plot=False)

    assert calls == [
        ("Q00", "X90"),
        ("Q00", "X180"),
        ("Q01", "X90"),
        ("Q01", "X180"),
    ]


def test_benchmark_2q_continues_after_failed_target() -> None:
    """Given one failed 2Q benchmark target, when running the suite, then later targets still run."""
    service = cast(Any, object.__new__(BenchmarkingService))
    calls: list[tuple[str, str]] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        cr_labels=["CR01", "CR23"]
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _interleaved_randomized_benchmarking(
        self: BenchmarkingService,
        targets: str,
        *,
        interleaved_clifford: str,
        **_kwargs: object,
    ) -> Result:
        calls.append((targets, interleaved_clifford))
        if targets == "CR01":
            raise RuntimeError("benchmark failed")
        return Result(data={targets: {"ok": True}})

    service.__dict__["interleaved_randomized_benchmarking"] = MethodType(
        _interleaved_randomized_benchmarking,
        service,
    )

    service.benchmark_2q(targets=["CR01", "CR23"], in_parallel=False, plot=False)

    assert calls == [("CR01", "ZX90"), ("CR23", "ZX90")]
