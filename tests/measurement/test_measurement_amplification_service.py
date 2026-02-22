"""Tests for measurement amplification service."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

from qubex.measurement.measurement_amplification_service import (
    MeasurementAmplificationService,
)


def test_apply_dc_voltages_resolves_targets_and_applies_voltages(monkeypatch) -> None:
    """Given targets, when applying DC voltages, then service applies mux-indexed voltages."""
    called: dict[str, Any] = {}

    @contextmanager
    def _fake_dc_voltage(voltages: dict[int, float]):
        called["voltages"] = voltages
        called["entered"] = True
        try:
            yield
        finally:
            called["exited"] = True

    monkeypatch.setattr(
        "qubex.measurement.measurement_amplification_service.dc_voltage",
        _fake_dc_voltage,
    )

    class _Mux:
        def __init__(self, index: int) -> None:
            self.index = index

    class _ControlParams:
        def get_dc_voltage(self, mux: int) -> float:
            return {0: 0.25, 2: -0.4}[mux]

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return {"Q00": "Q00", "RQ02": "Q02"}[target]

        def get_mux_by_qubit(self, qubit: str) -> _Mux:
            return {"Q00": _Mux(0), "Q02": _Mux(2)}[qubit]

    context = type("_Context", (), {"experiment_system": _ExperimentSystem()})()
    service = MeasurementAmplificationService(context=cast(Any, context))

    with service.apply_dc_voltages(["Q00", "RQ02"]):
        called["inside"] = True

    assert called["voltages"] == {1: 0.25, 3: -0.4}
    assert called["entered"] is True
    assert called["inside"] is True
    assert called["exited"] is True


def test_apply_dc_voltages_accepts_single_target(monkeypatch) -> None:
    """Given a single target string, when applying DC voltages, then service handles it as one target."""
    called: dict[str, Any] = {}

    @contextmanager
    def _fake_dc_voltage(voltages: dict[int, float]):
        called["voltages"] = voltages
        yield

    monkeypatch.setattr(
        "qubex.measurement.measurement_amplification_service.dc_voltage",
        _fake_dc_voltage,
    )

    class _Mux:
        def __init__(self, index: int) -> None:
            self.index = index

    class _ControlParams:
        def get_dc_voltage(self, mux: int) -> float:
            return {0: 0.25}[mux]

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return {"Q00": "Q00"}[target]

        def get_mux_by_qubit(self, qubit: str) -> _Mux:
            return {"Q00": _Mux(0)}[qubit]

    context = type("_Context", (), {"experiment_system": _ExperimentSystem()})()
    service = MeasurementAmplificationService(context=cast(Any, context))

    with service.apply_dc_voltages("Q00"):
        pass

    assert called["voltages"] == {1: 0.25}
