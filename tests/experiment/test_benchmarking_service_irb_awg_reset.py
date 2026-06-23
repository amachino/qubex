"""Tests for AWG reset behavior in interleaved randomized benchmarking."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np

from qubex.clifford.clifford import Clifford
from qubex.experiment.models.result import Result
from qubex.experiment.services.benchmarking_service import BenchmarkingService
from qubex.system import TargetType


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
    return _rb_payload_with_range(targets, np.array([0, 1], dtype=int))


def _rb_payload_with_range(targets: list[str], n_cliffords: np.ndarray) -> Result:
    """Build minimal RB-like payload with a specific Clifford sweep range."""
    mean = np.linspace(1.0, 0.9, len(n_cliffords), dtype=float)
    std = np.linspace(0.0, 0.01, len(n_cliffords), dtype=float)
    return Result(
        data={
            target: {
                "n_cliffords": n_cliffords,
                "mean": mean,
                "std": std,
                "trials": np.column_stack((mean, mean - 0.01)),
                "seeds": np.array([11, 22], dtype=int),
            }
            for target in targets
        }
    )


def _target(
    *,
    is_2q: bool,
    is_bswap: bool = False,
    target_type: TargetType | None = None,
) -> SimpleNamespace:
    """Build a target stub with the target-type flags used by benchmarking."""
    if target_type is None:
        target_type = TargetType.CTRL_CR if is_2q else TargetType.CTRL_GE
    return SimpleNamespace(
        is_2q=is_2q,
        is_bswap=is_bswap,
        type=target_type,
    )


def test_irb_experiment_resets_awg_once_for_1q(monkeypatch: Any) -> None:
    """Given 1Q IRB, when executed, then AWG reset runs once and inner RB calls skip reset."""
    _patch_fit_helpers(monkeypatch)

    reset_calls: list[set[str]] = []
    reset_flags: list[bool | None] = []
    time_integration_flags: list[bool | None] = []

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: _target(is_2q=False)
        ),
        resolve_qubit_label=lambda label: "Q17" if label == "custom-target" else label,
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(cliffords={"X90": Clifford.X90()})
    }

    def _rb_experiment_1q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        reset_flags.append(kwargs.get("reset_awg_and_capunits"))  # type: ignore[arg-type]
        time_integration_flags.append(kwargs.get("time_integration"))  # type: ignore[arg-type]
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
    assert time_integration_flags == [True, True]


def test_irb_experiment_resets_awg_once_for_2q(monkeypatch: Any) -> None:
    """Given 2Q IRB, when executed, then AWG reset runs once with both qubits and inner RB skips reset."""
    _patch_fit_helpers(monkeypatch)

    reset_calls: list[set[str]] = []
    reset_flags: list[bool | None] = []
    time_integration_flags: list[bool | None] = []

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: _target(is_2q=True)
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(cliffords={"II": Clifford.II()})
    }

    def _rb_experiment_2q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        reset_flags.append(kwargs.get("reset_awg_and_capunits"))  # type: ignore[arg-type]
        time_integration_flags.append(kwargs.get("time_integration"))  # type: ignore[arg-type]
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
    assert time_integration_flags == [True, True]


def test_irb_experiment_reuses_auto_reference_sweep_for_1q(
    monkeypatch: Any,
) -> None:
    """Given auto 1Q IRB, when reference stops early, then interleaved uses the reference sweep."""
    _patch_fit_helpers(monkeypatch)

    calls: list[np.ndarray | None] = []
    reference_range = np.array([0, 1, 2, 4], dtype=int)

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: _target(is_2q=False)
        ),
        resolve_qubit_label=lambda label: label,
        reset_awg_and_capunits=lambda *, qubits: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(cliffords={"X90": Clifford.X90()})
    }

    def _rb_experiment_1q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        labels = [targets] if isinstance(targets, str) else list(targets)
        n_cliffords_range = cast(np.ndarray | None, kwargs.get("n_cliffords_range"))
        calls.append(n_cliffords_range)
        if kwargs.get("interleaved_clifford") is None:
            return _rb_payload_with_range(labels, reference_range)
        if n_cliffords_range is None:
            return _rb_payload(labels)
        return _rb_payload_with_range(labels, n_cliffords_range)

    service.__dict__["rb_experiment_1q"] = MethodType(_rb_experiment_1q, service)

    service.irb_experiment(
        targets=["Q17"],
        interleaved_clifford="X90",
        plot=False,
        save_image=False,
    )

    assert calls[0] is None
    assert calls[1] is not None
    np.testing.assert_array_equal(calls[1], reference_range)


def test_irb_experiment_reuses_auto_reference_sweep_for_2q(
    monkeypatch: Any,
) -> None:
    """Given auto 2Q IRB, when reference stops early, then interleaved uses the reference sweep."""
    _patch_fit_helpers(monkeypatch)

    calls: list[np.ndarray | None] = []
    reference_range = np.array([0, 2, 4], dtype=int)

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: _target(is_2q=True)
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
        reset_awg_and_capunits=lambda *, qubits: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(cliffords={"II": Clifford.II()})
    }

    def _rb_experiment_2q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        labels = [targets] if isinstance(targets, str) else list(targets)
        n_cliffords_range = cast(np.ndarray | None, kwargs.get("n_cliffords_range"))
        calls.append(n_cliffords_range)
        if kwargs.get("interleaved_clifford") is None:
            return _rb_payload_with_range(labels, reference_range)
        if n_cliffords_range is None:
            return _rb_payload(labels)
        return _rb_payload_with_range(labels, n_cliffords_range)

    service.__dict__["rb_experiment_2q"] = MethodType(_rb_experiment_2q, service)

    service.irb_experiment(
        targets=["CR17-18"],
        interleaved_clifford="II",
        plot=False,
        save_image=False,
    )

    assert calls[0] is None
    assert calls[1] is not None
    np.testing.assert_array_equal(calls[1], reference_range)


def test_irb_experiment_includes_reference_and_interleaved_trial_data(
    monkeypatch: Any,
) -> None:
    """Given IRB, final result includes per-trial reference and interleaved curve data."""
    _patch_fit_helpers(monkeypatch)

    reference_range = np.array([0, 1, 2], dtype=int)
    interleaved_range = np.array([0, 1, 2], dtype=int)

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: _target(is_2q=False)
        ),
        resolve_qubit_label=lambda label: label,
        reset_awg_and_capunits=lambda *, qubits: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(cliffords={"X90": Clifford.X90()})
    }

    def _rb_experiment_1q(
        self: BenchmarkingService,
        targets: list[str] | str,
        **kwargs: object,
    ) -> Result:
        labels = [targets] if isinstance(targets, str) else list(targets)
        if kwargs.get("interleaved_clifford") is None:
            return _rb_payload_with_range(labels, reference_range)
        return _rb_payload_with_range(labels, interleaved_range)

    service.__dict__["rb_experiment_1q"] = MethodType(_rb_experiment_1q, service)

    result = service.irb_experiment(
        targets=["Q17"],
        interleaved_clifford="X90",
        n_trials=2,
        plot=False,
        save_image=False,
    )

    rb_data = result["Q17"]["rb_data"]
    irb_data = result["Q17"]["irb_data"]
    np.testing.assert_array_equal(rb_data["n_cliffords"], reference_range)
    np.testing.assert_array_equal(irb_data["n_cliffords"], interleaved_range)
    np.testing.assert_array_equal(rb_data["seeds"], np.array([11, 22], dtype=int))
    np.testing.assert_array_equal(irb_data["seeds"], np.array([11, 22], dtype=int))
    assert rb_data["trials"].shape == (3, 2)
    assert irb_data["trials"].shape == (3, 2)
