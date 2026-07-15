"""Tests for Experiment JPA DC voltage helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.experiment.experiment_context import ExperimentContext


@dataclass(frozen=True)
class _Mux:
    index: int
    label: str


class _ControlParams:
    def get_dc_voltage(self, mux: int) -> float:
        return {6: 0.76, 7: 0.42}[mux]


class _ExperimentSystem:
    control_params = _ControlParams()

    def __init__(self) -> None:
        self.muxes = {
            "MUX06": _Mux(index=6, label="MUX06"),
            "MUX07": _Mux(index=7, label="MUX07"),
        }

    def get_mux(self, mux: int | str) -> _Mux:
        if isinstance(mux, int):
            return {item.index: item for item in self.muxes.values()}[mux]
        return self.muxes[mux]


class _DCVoltageController:
    def __init__(self) -> None:
        self.voltages: dict[int, float] = {}
        self.output_states: dict[int, int] = {}
        self.calls: list[tuple[object, ...]] = []

    def on(self, *, channel: int) -> None:
        self.calls.append(("on", channel))
        self.output_states[channel] = 1

    def off(self, *, channel: int) -> None:
        self.calls.append(("off", channel))
        self.output_states[channel] = 0

    def set_voltage(self, *, channel: int, voltage: float) -> None:
        self.calls.append(("set_voltage", channel, voltage))
        self.voltages[channel] = voltage

    def get_voltage(self, *, channel: int) -> float:
        self.calls.append(("get_voltage", channel))
        return self.voltages[channel]

    def get_output_state(self, *, channel: int) -> int:
        self.calls.append(("get_output_state", channel))
        return self.output_states[channel]


class _SystemManager:
    def __init__(self, dc_controller: _DCVoltageController) -> None:
        self.dc_voltage_controller = dc_controller
        self.experiment_system = _ExperimentSystem()


class _ContextForTest(ExperimentContext):
    def __init__(
        self,
        *,
        mux_labels: list[str],
        dc_controller: _DCVoltageController,
    ) -> None:
        self._mux_labels_for_test = mux_labels
        self._system_manager_for_test = _SystemManager(dc_controller)

    @property
    def mux_labels(self) -> list[str]:
        return self._mux_labels_for_test

    @property
    def system_manager(self) -> _SystemManager:
        return self._system_manager_for_test


def test_set_jpa_dc_voltage_uses_default_active_mux_and_param_voltage() -> None:
    """Given one active mux, setting JPA DC should use configured default voltage."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    ctx.set_jpa_dc_voltage()

    assert dc_controller.voltages[7] == pytest.approx(0.76)
    assert dc_controller.output_states[7] == 1


def test_set_jpa_dc_voltage_accepts_explicit_mux_and_voltage() -> None:
    """Given explicit mux and voltage, setting JPA DC should use that channel."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    ctx.set_jpa_dc_voltage(0.5, mux=7)

    assert dc_controller.voltages[8] == pytest.approx(0.5)
    assert dc_controller.output_states[8] == 1


def test_set_jpa_dc_voltage_requires_mux_when_multiple_are_active() -> None:
    """Given multiple active muxes, omitted mux should raise."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    with pytest.raises(ValueError, match="multiple active muxes"):
        ctx.set_jpa_dc_voltage()


def test_jpa_dc_voltage_context_turns_off_on_exit() -> None:
    """Given JPA DC context, exiting should turn off the selected channel."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.jpa_dc_voltage(0.5):
        assert dc_controller.output_states[7] == 1

    assert dc_controller.output_states[7] == 0
