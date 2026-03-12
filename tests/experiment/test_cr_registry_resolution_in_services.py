"""Tests for CR label resolution through experiment context in services."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from qxpulse import PulseArray, PulseSchedule

from qubex.experiment.services.benchmarking_service import BenchmarkingService
from qubex.experiment.services.calibration_service import CalibrationService


def test_calibrate_2q_uses_context_cr_pair_resolution() -> None:
    """Given custom CR labels, when calibrating 2Q gates, then pair resolution uses context mapping."""
    service = cast(Any, object.__new__(CalibrationService))
    captured: list[tuple[str, str]] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        cr_labels=["CR_CUSTOM"],
        cr_pair=lambda _label: ("Q17", "Q18"),
        save_calib_note=lambda: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["obtain_cr_params"] = lambda **kwargs: (
        captured.append((kwargs["control_qubit"], kwargs["target_qubit"]))
        or {"ok": True}
    )
    service.__dict__["calibrate_zx90"] = lambda **kwargs: {"ok": kwargs}

    service.calibrate_2q(targets=["CR_CUSTOM"], plot=False)

    assert captured == [("Q17", "Q18")]


def test_rb_sequence_2q_uses_context_cr_pair_resolution() -> None:
    """Given custom CR labels, when building 2Q RB sequence, then pair resolution uses context mapping."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=True)
        ),
        cr_pair=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
        zx90=lambda control, target: PulseSchedule([control, target]),
    )
    service.__dict__["_clifford_generator"] = SimpleNamespace(
        create_rb_sequences=lambda **_kwargs: ([], [])
    )

    schedule = service.rb_sequence_2q(target="CR_CUSTOM", n=0)

    assert list(schedule.labels) == ["Q17", "CR_CUSTOM", "Q18"]
