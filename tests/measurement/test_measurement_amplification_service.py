"""Tests for measurement amplification service."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

from qubex.external_devices import DCVoltageProfile
from qubex.measurement.services.measurement_amplification_service import (
    MeasurementAmplificationService,
)


def test_apply_dc_voltages_resolves_targets_and_applies_voltages(monkeypatch) -> None:
    """Given targets, when applying DC voltages, then service applies mux-indexed voltages."""
    called: dict[str, Any] = {}

    class _Mux:
        def __init__(self, index: int) -> None:
            self.index = index

    class _ControlParams:
        def has_optimal_voltage(self, mux: int) -> bool:
            return True

        def get_optimal_voltage(self, mux: int) -> float:
            return {0: 0.25, 2: -0.4}[mux]

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return {"Q00": "Q00", "RQ02": "Q02"}[target]

        def get_mux_by_qubit(self, qubit: str) -> _Mux:
            return {"Q00": _Mux(0), "Q02": _Mux(2)}[qubit]

    class _DCVoltageController:
        @contextmanager
        def apply_voltages(
            self,
            requests: dict[int, tuple[float, DCVoltageProfile]],
        ):
            called["requests"] = requests
            called["entered"] = True
            try:
                yield
            finally:
                called["exited"] = True

    system_manager = type(
        "_SystemManager",
        (),
        {
            "dc_voltage_controller": _DCVoltageController(),
            "resolve_dc_voltage_profile": staticmethod(
                {
                    0: DCVoltageProfile(channel=2),
                    2: DCVoltageProfile(channel=4, ramp_rate_v_per_s=0.05),
                }.__getitem__
            ),
        },
    )()
    context = type(
        "_Context",
        (),
        {
            "experiment_system": _ExperimentSystem(),
            "system_manager": system_manager,
        },
    )()
    service = MeasurementAmplificationService(context=cast(Any, context))

    with service.apply_dc_voltages(["Q00", "RQ02"]):
        called["inside"] = True

    assert called["requests"] == {
        2: (0.25, DCVoltageProfile(channel=2)),
        4: (-0.4, DCVoltageProfile(channel=4, ramp_rate_v_per_s=0.05)),
    }
    assert called["entered"] is True
    assert called["inside"] is True
    assert called["exited"] is True


def test_apply_dc_voltages_accepts_single_target(monkeypatch) -> None:
    """Given a single target string, when applying DC voltages, then service handles it as one target."""
    called: dict[str, Any] = {}

    class _Mux:
        def __init__(self, index: int) -> None:
            self.index = index

    class _ControlParams:
        def has_optimal_voltage(self, mux: int) -> bool:
            return True

        def get_optimal_voltage(self, mux: int) -> float:
            return {0: 0.25}[mux]

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return {"Q00": "Q00"}[target]

        def get_mux_by_qubit(self, qubit: str) -> _Mux:
            return {"Q00": _Mux(0)}[qubit]

    class _DCVoltageController:
        @contextmanager
        def apply_voltages(
            self,
            requests: dict[int, tuple[float, DCVoltageProfile]],
        ):
            called["requests"] = requests
            yield

    system_manager = type(
        "_SystemManager",
        (),
        {
            "dc_voltage_controller": _DCVoltageController(),
            "resolve_dc_voltage_profile": staticmethod(
                {0: DCVoltageProfile(channel=2)}.__getitem__
            ),
        },
    )()
    context = type(
        "_Context",
        (),
        {
            "experiment_system": _ExperimentSystem(),
            "system_manager": system_manager,
        },
    )()
    service = MeasurementAmplificationService(context=cast(Any, context))

    with service.apply_dc_voltages("Q00"):
        pass

    assert called["requests"] == {2: (0.25, DCVoltageProfile(channel=2))}


def test_apply_dc_voltages_skips_uncalibrated_muxes() -> None:
    """Muxes without a calibrated optimal voltage should not touch the controller."""

    class _ControlParams:
        def has_optimal_voltage(self, mux: int) -> bool:
            return False

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return target

        def get_mux_by_qubit(self, qubit: str):
            return type("_Mux", (), {"index": 0})()

    class _Controller:
        @contextmanager
        def apply_voltages(self, requests):
            raise AssertionError("controller must not be used")
            yield

    system_manager = type(
        "_SystemManager",
        (),
        {"dc_voltage_controller": _Controller()},
    )()
    context = type(
        "_Context",
        (),
        {
            "experiment_system": _ExperimentSystem(),
            "system_manager": system_manager,
        },
    )()
    service = MeasurementAmplificationService(context=cast(Any, context))

    with service.apply_dc_voltages("Q00"):
        pass
