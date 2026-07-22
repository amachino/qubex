"""Tests for Experiment DC voltage helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.models import DCVoltageState


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
        self.output_states: dict[int, bool] = {}
        self.calls: list[tuple[object, ...]] = []
        self.fail_set_voltage = False

    def on(self, *, channel: int) -> None:
        self.calls.append(("on", channel))
        self.output_states[channel] = True

    def off(self, *, channel: int) -> None:
        self.calls.append(("off", channel))
        self.output_states[channel] = False

    def set_voltage(self, *, channel: int, voltage: float) -> None:
        if self.fail_set_voltage:
            raise RuntimeError("restore failed")
        self.calls.append(("set_voltage", channel, voltage))
        self.voltages[channel] = voltage

    def get_voltage(self, *, channel: int) -> float:
        self.calls.append(("get_voltage", channel))
        return self.voltages[channel]

    def is_output_on(self, *, channel: int) -> bool:
        self.calls.append(("is_output_on", channel))
        return self.output_states[channel]


class _SystemManager:
    def __init__(
        self,
        dc_controller: _DCVoltageController,
        mux_to_channel: dict[int, int] | None = None,
    ) -> None:
        self.dc_voltage_controller = dc_controller
        self.experiment_system = _ExperimentSystem()
        self.mux_to_channel = mux_to_channel or {}

    def resolve_dc_voltage_channel(self, mux_index: int) -> int:
        return self.mux_to_channel.get(mux_index, mux_index + 1)


class _ContextForTest(ExperimentContext):
    def __init__(
        self,
        *,
        mux_labels: list[str],
        dc_controller: _DCVoltageController,
        mux_to_channel: dict[int, int] | None = None,
    ) -> None:
        self._mux_labels_for_test = mux_labels
        self._system_manager_for_test = _SystemManager(
            dc_controller,
            mux_to_channel,
        )

    @property
    def mux_labels(self) -> list[str]:
        return self._mux_labels_for_test

    @property
    def system_manager(self) -> _SystemManager:
        return self._system_manager_for_test


def test_dc_voltage_control_uses_configured_channel_mapping() -> None:
    """Given a mapped mux, DC control should use and turn off the mapped channel."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06"],
        dc_controller=dc_controller,
        mux_to_channel={6: 1},
    )

    with ctx.dc_voltage_control() as dc:
        dc.set_voltage(0.5)
        assert dc_controller.voltages[1] == pytest.approx(0.5)
        assert dc_controller.output_states[1] is True

    assert dc_controller.output_states[1] is False


@pytest.mark.parametrize(
    ("controller_state", "output", "is_on"),
    [(False, "off", False), (True, "on", True)],
)
def test_dc_voltage_control_state_returns_mapped_controller_readback(
    controller_state: bool,
    output: str,
    is_on: bool,
) -> None:
    """Given a mapped mux, DC state should include its channel and readback."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages[1] = 0.54
    dc_controller.output_states[1] = controller_state
    ctx = _ContextForTest(
        mux_labels=["MUX06"],
        dc_controller=dc_controller,
        mux_to_channel={6: 1},
    )

    with ctx.dc_voltage_control() as dc:
        state = dc.state

    assert state == DCVoltageState(
        mux_label="MUX06",
        mux_index=6,
        channel=1,
        voltage=pytest.approx(0.54),
        output=output,
    )
    assert state.is_on is is_on


def test_dc_voltage_control_requires_mux_when_multiple_are_active() -> None:
    """Given multiple active muxes, omitted mux should raise."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    with (
        pytest.raises(ValueError, match="multiple active muxes"),
        ctx.dc_voltage_control(),
    ):
        pass


def test_dc_voltage_control_sweeps_states_and_turns_off_on_exit() -> None:
    """Given a DC control, sweep should yield readback states and turn off on exit."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        states = list(dc.sweep(sweep_range=[0.1, 0.2]))

        assert [state.voltage for state in states] == pytest.approx([0.1, 0.2])
        assert all(state.output == "on" for state in states)
        assert dc_controller.output_states[7] is True

    assert dc_controller.output_states[7] is False
    assert dc_controller.voltages[7] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("start", "target", "expected"),
    [
        (0.0, 0.25, [0.0, 0.1, 0.2, 0.25]),
        (0.3, 0.05, [0.3, 0.2, 0.1, 0.05]),
    ],
)
def test_dc_voltage_control_ramps_to_target_from_selected_direction(
    start: float,
    target: float,
    expected: list[float],
) -> None:
    """Given a ramp path, control should approach and include the exact target."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        state = dc.ramp_to(target, start=start, step=0.1)
        applied = [call[2] for call in dc_controller.calls if call[0] == "set_voltage"]
        assert applied == pytest.approx(expected)

    assert state.voltage == pytest.approx(target)


def test_dc_voltage_control_restores_ramp_start_before_turning_off() -> None:
    """Given a ramped control, context exit should ramp back before output off."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        dc.ramp_to(0.25, start=0.0, step=0.1)

    applied = [call[2] for call in dc_controller.calls if call[0] == "set_voltage"]
    assert applied == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.15, 0.05, 0.0])
    assert dc_controller.calls[-1] == ("off", 7)


def test_dc_voltage_control_can_skip_ramp_restore_on_exit() -> None:
    """Given restore disabled, context exit should turn output off immediately."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        dc.ramp_to(
            0.2,
            start=0.0,
            step=0.1,
            restore_on_exit=False,
        )

    applied = [call[2] for call in dc_controller.calls if call[0] == "set_voltage"]
    assert applied == pytest.approx([0.0, 0.1, 0.2])
    assert dc_controller.calls[-1] == ("off", 7)


def test_dc_voltage_control_turns_off_when_ramp_restore_fails() -> None:
    """Given a restore failure, context exit should still turn the output off."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    def exit_with_failed_restore() -> None:
        with ctx.dc_voltage_control() as dc:
            dc.ramp_to(0.2, start=0.0, step=0.1)
            dc_controller.fail_set_voltage = True

    with pytest.raises(RuntimeError, match="restore failed"):
        exit_with_failed_restore()

    assert dc_controller.calls[-1] == ("off", 7)


def test_dc_voltage_control_ramp_waits_after_every_setpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a ramp delay, control should settle after every applied setpoint."""
    delays: list[float] = []
    monkeypatch.setattr("qubex.experiment.dc_voltage_control.time.sleep", delays.append)
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        dc.ramp_to(0.2, start=0.0, step=0.1, delay=0.05)
        assert delays == [0.05, 0.05, 0.05]

    assert delays == [0.05, 0.05, 0.05, 0.05, 0.05]


@pytest.mark.parametrize("step", [0.0, -0.1])
def test_dc_voltage_control_ramp_rejects_non_positive_step(step: float) -> None:
    """Given a non-positive step, ramping should reject the ambiguous path."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with (
        ctx.dc_voltage_control() as dc,
        pytest.raises(ValueError, match="step must be positive"),
    ):
        dc.ramp_to(0.2, start=0.0, step=step)


def test_dc_voltage_control_sweep_waits_after_each_setpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a sweep delay, DC control should wait after setting every voltage."""
    delays: list[float] = []
    monkeypatch.setattr("qubex.experiment.dc_voltage_control.time.sleep", delays.append)
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        list(dc.sweep(sweep_range=[0.1, 0.2], delay=0.05))

    assert delays == [0.05, 0.05]


def test_dc_voltage_control_turns_off_after_partial_sweep() -> None:
    """Given a partial DC sweep, exiting its control should turn off the output."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        states = dc.sweep(sweep_range=[0.1, 0.2])
        assert next(states).voltage == pytest.approx(0.1)

    assert dc_controller.output_states[7] is False


def test_dc_voltage_control_sets_and_reads_bound_mux() -> None:
    """Given a bound DC control, set and state should use the selected mux."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    with ctx.dc_voltage_control(mux=7) as dc:
        state = dc.set_voltage(0.4)

        assert state == dc.state
        assert state.mux_label == "MUX07"
        assert state.channel == 8
        assert state.voltage == pytest.approx(0.4)
        assert state.is_on

    assert dc_controller.output_states[8] is False


def test_dc_voltage_control_stops_after_repeated_readback_mismatch() -> None:
    """Given persistent readback error, bound control should fail without hanging."""

    class _MismatchedReadbackController(_DCVoltageController):
        def get_voltage(self, *, channel: int) -> float:
            self.calls.append(("get_voltage", channel))
            return -1.0

    dc_controller = _MismatchedReadbackController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with (
        ctx.dc_voltage_control() as dc,
        pytest.raises(RuntimeError, match="failed to reach"),
    ):
        dc.set_voltage(0.5)

    set_calls = [call for call in dc_controller.calls if call[0] == "set_voltage"]
    assert len(set_calls) == 3
