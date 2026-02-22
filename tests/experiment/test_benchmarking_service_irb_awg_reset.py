"""Tests for AWG reset behavior in interleaved randomized benchmarking."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np

from qubex.clifford.clifford import Clifford
from qubex.experiment.models.result import Result
from qubex.experiment.services.benchmarking_service import BenchmarkingService


def _patch_fit_helpers(monkeypatch: Any) -> None:
    """Patch fit helpers to keep IRB tests lightweight."""

    def _fit_rb(**_kwargs: object) -> dict[str, float]:
        return {
            "A": 0.1,
            "p": 0.99,
            "p_err": 0.01,
            "C": 0.9,
            "avg_gate_error": 0.01,
            "avg_gate_fidelity": 0.99,
            "avg_gate_fidelity_err": 0.001,
        }

    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.fit_rb",
        _fit_rb,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.benchmarking_service.fitting.plot_irb",
        lambda **_kwargs: object(),
    )


def _rb_payload(targets: list[str]) -> Result:
    """Build minimal RB-like payload consumed by IRB post-processing."""
    return Result(
        data={
            target: {
                "n_cliffords": np.array([0, 1], dtype=int),
                "mean": np.array([1.0, 0.9], dtype=float),
                "std": np.array([0.0, 0.01], dtype=float),
            }
            for target in targets
        }
    )


def test_irb_experiment_resets_awg_once_for_1q(monkeypatch: Any) -> None:
    """Given 1Q IRB, when executed, then AWG reset runs once and inner RB calls skip reset."""
    _patch_fit_helpers(monkeypatch)

    reset_calls: list[set[str]] = []
    reset_flags: list[bool | None] = []

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=False)
        ),
        resolve_qubit_label=lambda label: "Q17" if label == "custom-target" else label,
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator"] = SimpleNamespace(
        cliffords={"X90": Clifford.X90()}
    )

    def _rb_experiment_1q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        reset_flags.append(kwargs.get("reset_awg_and_capunits"))  # type: ignore[arg-type]
        labels = [targets] if isinstance(targets, str) else list(targets)
        return _rb_payload(labels)

    service.__dict__["rb_experiment_1q"] = MethodType(_rb_experiment_1q, service)

    service.irb_experiment(
        targets=["custom-target"],
        interleaved_clifford="X90",
        plot=False,
        save_image=False,
    )

    assert reset_calls == [{"Q17"}]
    assert reset_flags == [False, False]


def test_irb_experiment_resets_awg_once_for_2q(monkeypatch: Any) -> None:
    """Given 2Q IRB, when executed, then AWG reset runs once with both qubits and inner RB skips reset."""
    _patch_fit_helpers(monkeypatch)

    reset_calls: list[set[str]] = []
    reset_flags: list[bool | None] = []

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=True)
        ),
        cr_pair=lambda _label: ("Q17", "Q18"),
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator"] = SimpleNamespace(
        cliffords={"II": Clifford.II()}
    )

    def _rb_experiment_2q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        reset_flags.append(kwargs.get("reset_awg_and_capunits"))  # type: ignore[arg-type]
        labels = [targets] if isinstance(targets, str) else list(targets)
        return _rb_payload(labels)

    service.__dict__["rb_experiment_2q"] = MethodType(_rb_experiment_2q, service)

    service.irb_experiment(
        targets=["CR17-18"],
        interleaved_clifford="II",
        plot=False,
        save_image=False,
    )

    assert reset_calls == [{"Q17", "Q18"}]
    assert reset_flags == [False, False]
