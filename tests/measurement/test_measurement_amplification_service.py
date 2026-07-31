"""Tests for measurement amplification service."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

import pytest

from qubex.external_devices import (
    DCVoltageExitMode,
    DCVoltageExitPolicy,
    DCVoltageProfile,
)
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
        def get_dc_voltage(self, mux: int) -> float:
            return {0: 0.25, 2: -0.4}[mux]

        def get_dc_voltage_exit_mode(self, mux: int) -> str:
            return {0: "low_noise", 2: "off"}[mux]

        def get_low_noise_dc_voltage(self, mux: int) -> float:
            return {0: -0.08}[mux]

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
            *,
            exit_policies: dict[int, DCVoltageExitPolicy],
        ):
            called["requests"] = requests
            called["exit_policies"] = exit_policies
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
    assert called["exit_policies"] == {
        2: DCVoltageExitPolicy(
            mode=DCVoltageExitMode.TARGET,
            target_voltage_v=-0.08,
        ),
        4: DCVoltageExitPolicy(mode=DCVoltageExitMode.SHUTDOWN),
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
        def get_dc_voltage(self, mux: int) -> float:
            return {0: 0.25}[mux]

        def get_dc_voltage_exit_mode(self, mux: int) -> str:
            return "off"

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
            *,
            exit_policies: dict[int, DCVoltageExitPolicy],
        ):
            called["requests"] = requests
            called["exit_policies"] = exit_policies
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
    assert called["exit_policies"] == {
        2: DCVoltageExitPolicy(mode=DCVoltageExitMode.SHUTDOWN)
    }


@pytest.mark.parametrize(
    ("on_exit", "expected_policy"),
    [
        ("off", DCVoltageExitPolicy(mode=DCVoltageExitMode.SHUTDOWN)),
        ("hold", DCVoltageExitPolicy(mode=DCVoltageExitMode.HOLD)),
        ("restore", DCVoltageExitPolicy(mode=DCVoltageExitMode.RESTORE)),
        (
            "low_noise",
            DCVoltageExitPolicy(
                mode=DCVoltageExitMode.TARGET,
                target_voltage_v=-0.08,
            ),
        ),
    ],
)
def test_apply_dc_voltages_allows_exit_mode_override(
    on_exit: str,
    expected_policy: DCVoltageExitPolicy,
) -> None:
    """An API exit mode should override the mux default."""
    called: dict[str, Any] = {}

    class _ControlParams:
        def get_dc_voltage(self, mux: int) -> float:
            return 0.25

        def get_dc_voltage_exit_mode(self, mux: int) -> str:
            return "off"

        def get_low_noise_dc_voltage(self, mux: int) -> float:
            return -0.08

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return target

        def get_mux_by_qubit(self, qubit: str):
            return type("_Mux", (), {"index": 0})()

    class _Controller:
        @contextmanager
        def apply_voltages(self, requests, *, exit_policies):
            called["exit_policies"] = exit_policies
            yield

    system_manager = type(
        "_SystemManager",
        (),
        {
            "dc_voltage_controller": _Controller(),
            "resolve_dc_voltage_profile": staticmethod(
                lambda mux: DCVoltageProfile(channel=2)
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

    with service.apply_dc_voltages("Q00", on_exit=on_exit):
        pass

    assert called["exit_policies"] == {2: expected_policy}


def test_apply_dc_voltages_requires_low_noise_voltage() -> None:
    """Low-noise exit should fail when its mux has no calibrated voltage."""

    class _ControlParams:
        def get_dc_voltage(self, mux: int) -> float:
            return 0.25

        def get_dc_voltage_exit_mode(self, mux: int) -> str:
            return "low_noise"

        def get_low_noise_dc_voltage(self, mux: int) -> float:
            raise ValueError("No low-noise DC voltage")

    class _ExperimentSystem:
        control_params = _ControlParams()

        def resolve_qubit_label(self, target: str) -> str:
            return target

        def get_mux_by_qubit(self, qubit: str):
            return type("_Mux", (), {"index": 0})()

    system_manager = type(
        "_SystemManager",
        (),
        {
            "resolve_dc_voltage_profile": staticmethod(
                lambda mux: DCVoltageProfile(channel=2)
            )
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

    with (
        pytest.raises(ValueError, match="low-noise DC voltage"),
        service.apply_dc_voltages("Q00"),
    ):
        pass
