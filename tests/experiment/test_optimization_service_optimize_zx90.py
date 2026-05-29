"""Tests for ZX90 optimization parameter update behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import qubex.experiment.services.optimization_service as optimization_module
from qubex.experiment.services.optimization_service import OptimizationService


def _cr_param() -> dict[str, float | str]:
    return {
        "target": "Q28-Q25",
        "duration": 96.0,
        "ramptime": 16.0,
        "cr_amplitude": 0.624636,
        "cr_phase": 1.024889,
        "cr_beta": 0.218219,
        "cancel_amplitude": 0.019198,
        "cancel_phase": -3.087765,
        "cancel_beta": 0.0,
        "rotary_amplitude": 0.026342,
        "zx_rotation_rate": 0.0019369457330069555,
        "timestamp": "2026-05-29 14:02:43",
    }


def _make_service(
    monkeypatch: Any, stored: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []

    def update_cr_param(_target: str, value: dict[str, Any]) -> None:
        updates.append(dict(value))
        stored.update(value)

    service = cast(Any, object.__new__(OptimizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        calib_note=SimpleNamespace(
            get_cr_param=lambda _target: stored,
            update_cr_param=update_cr_param,
        )
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x180=lambda _target: object(),
    )
    service.__dict__["_benchmarking_service"] = SimpleNamespace(
        rb_sequence_2q=lambda *_args, **_kwargs: object(),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        measure=lambda *_args, **_kwargs: SimpleNamespace(
            get_mitigated_probabilities=lambda _targets: {"00": 0.9}
        )
    )

    class FakeCMAEvolutionStrategy:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.result = SimpleNamespace(xbest=np.array([0.7]))

        def optimize(self, objective_func: Any) -> None:
            objective_func(self.result.xbest)

    monkeypatch.setattr(
        optimization_module,
        "_load_cma",
        lambda: SimpleNamespace(CMAEvolutionStrategy=FakeCMAEvolutionStrategy),
    )
    monkeypatch.setattr(
        optimization_module,
        "CrossResonance",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return service, updates


def test_optimize_zx90_update_false_does_not_mutate_calib_note(
    monkeypatch: Any,
) -> None:
    """Given update disabled, optimizing returns candidate params without changing stored CR params."""
    stored = _cr_param()
    service, updates = _make_service(monkeypatch, stored)

    result = service.optimize_zx90(
        "Q28",
        "Q25",
        objective_type="rb",
        optimize_method="cma",
        update_cr_param=False,
        opt_params=["cr_amplitude"],
        n_trials=1,
        maxiter=1,
    )

    assert stored["cr_amplitude"] == 0.624636
    assert result["cr_param"]["cr_amplitude"] == 0.7
    assert updates == []


def test_optimize_zx90_update_true_updates_calib_note(
    monkeypatch: Any,
) -> None:
    """Given update enabled, optimizing writes candidate params back to calibration note memory."""
    stored = _cr_param()
    service, updates = _make_service(monkeypatch, stored)

    result = service.optimize_zx90(
        "Q28",
        "Q25",
        objective_type="rb",
        optimize_method="cma",
        update_cr_param=True,
        opt_params=["cr_amplitude"],
        n_trials=1,
        maxiter=1,
    )

    assert stored["cr_amplitude"] == 0.7
    assert result["cr_param"]["cr_amplitude"] == 0.7
    assert updates[0]["cr_amplitude"] == 0.7
