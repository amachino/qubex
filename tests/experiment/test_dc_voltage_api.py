"""Tests for Experiment DC voltage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from qubex.experiment.experiment_context import ExperimentContext
from qubex.external_devices import DCVoltageProfile


@dataclass(frozen=True)
class _Mux:
    index: int
    label: str


class _ControlParams:
    def __init__(self, bias_voltages: dict[int, float] | None = None) -> None:
        self.bias_voltages = (
            bias_voltages if bias_voltages is not None else {6: 0.76, 7: 0.42}
        )

    def has_bias_voltage(self, mux: int) -> bool:
        return mux in self.bias_voltages

    def get_bias_voltage(self, mux: int) -> float:
        if mux not in self.bias_voltages:
            raise ValueError(
                f"Mux {mux} has no calibrated `bias_voltage` in `jpa_params.yaml`."
            )
        return self.bias_voltages[mux]


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
        return self.voltages.get(channel, 0.0)

    def is_output_on(self, *, channel: int) -> bool:
        self.calls.append(("is_output_on", channel))
        return self.output_states.get(channel, False)

    def apply_voltage(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        self.calls.append(("apply_voltage", channel, voltage, profile))
        self.voltages[channel] = voltage
        self.output_states[channel] = True

    def apply_voltage_immediately(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        self.calls.append(("apply_voltage_immediately", channel, voltage, profile))
        self.voltages[channel] = voltage
        self.output_states[channel] = True

    def idle(self, *, channel: int, profile: DCVoltageProfile) -> None:
        self.calls.append(("idle", channel, profile))
        if self.fail_set_voltage:
            raise RuntimeError("restore failed")
        self.voltages[channel] = profile.idle_voltage_v

    def apply_channels(
        self, requests: dict[int, tuple[float, DCVoltageProfile]]
    ) -> None:
        self.calls.append(("apply_channels", tuple(requests)))
        for channel, (voltage, _) in requests.items():
            self.voltages[channel] = voltage
            self.output_states[channel] = True

    def reset_channels(self, profiles: dict[int, DCVoltageProfile]) -> None:
        self.calls.append(("reset_channels", tuple(profiles)))
        for channel, profile in profiles.items():
            self.voltages[channel] = profile.reset_voltage_v
            self.output_states[channel] = True

    def turn_off_channels(self, profiles: dict[int, DCVoltageProfile]) -> None:
        self.calls.append(("turn_off_channels", tuple(profiles)))
        for channel in profiles:
            self.voltages[channel] = 0.0
            self.output_states[channel] = False

    def read_channels(self, channels: list[int]) -> dict[int, tuple[float, bool]]:
        self.calls.append(("read_channels", tuple(channels)))
        return {
            channel: (
                self.voltages.get(channel, 0.0),
                self.output_states.get(channel, False),
            )
            for channel in channels
        }

    def idle_channels(self, profiles: dict[int, DCVoltageProfile]) -> None:
        self.calls.append(("idle_channels", tuple(profiles)))
        for channel, profile in profiles.items():
            self.voltages[channel] = profile.idle_voltage_v


class _SystemManager:
    def __init__(
        self,
        dc_controller: _DCVoltageController,
        mux_to_channel: dict[int, int] | None = None,
    ) -> None:
        self.dc_voltage_controller = dc_controller
        self.experiment_system = _ExperimentSystem()
        self.mux_to_channel = mux_to_channel or {}

    def dc_voltage_mux_indices(self) -> list[int]:
        return sorted(self.mux_to_channel) if self.mux_to_channel else [6, 7]

    def resolve_dc_voltage_channel(self, mux_index: int) -> int:
        return self.mux_to_channel.get(mux_index, mux_index + 1)

    def resolve_dc_voltage_profile(self, mux_index: int) -> DCVoltageProfile:
        return DCVoltageProfile(
            channel=self.resolve_dc_voltage_channel(mux_index),
            ramp_rate_v_per_s=1.0,
            ramp_step_size_v=0.1,
            ramp_wait_s=0.1,
            idle_voltage_v=0.0,
            readback_tolerance_v=0.002,
        )


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
    """Given a mapped mux, DC control should drive and idle the mapped channel."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06"],
        dc_controller=dc_controller,
        mux_to_channel={6: 1},
    )

    with ctx.dc_voltage_control() as dc:
        dc.apply_voltage(0.5)
        assert dc_controller.voltages[1] == pytest.approx(0.5)
        assert dc_controller.output_states[1] is True

    assert dc_controller.voltages[1] == pytest.approx(0.0)
    assert dc_controller.output_states[1] is True


@pytest.mark.parametrize(
    ("controller_state", "output", "is_on"),
    [(False, "off", False), (True, "on", True)],
)
def test_dc_voltage_control_state_returns_mapped_controller_readback(
    controller_state: bool,
    output: Literal["on", "off"],
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

    assert state.mux_label == "MUX06"
    assert state.mux_index == 6
    assert state.channel == 1
    assert state.voltage == pytest.approx(0.54)
    assert state.output == output
    assert state.is_on is is_on


def test_get_dc_voltage_state_reads_without_changing_output() -> None:
    """DC state retrieval should read the mapped output without changing it."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages[1] = 0.54
    dc_controller.output_states[1] = True
    ctx = _ContextForTest(
        mux_labels=["MUX06"],
        dc_controller=dc_controller,
        mux_to_channel={6: 1},
    )

    state = ctx.get_dc_voltage_state(mux=6)

    assert state.voltage == pytest.approx(0.54)
    assert state.output == "on"
    assert dc_controller.calls == [("read_channels", (1,))]


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


def test_dc_voltage_control_sweeps_states_and_idles_on_exit() -> None:
    """Given a DC control, sweep should yield readback states and idle on exit."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        states = list(dc.sweep(sweep_range=[0.1, 0.2]))

        assert [state.voltage for state in states] == pytest.approx([0.1, 0.2])
        assert all(state.output == "on" for state in states)
        assert dc_controller.output_states[7] is True

    assert dc_controller.voltages[7] == pytest.approx(0.0)
    assert dc_controller.output_states[7] is True


def test_dc_voltage_control_applies_voltage_with_configured_ramp() -> None:
    """Applying voltage should delegate the configured profile to the controller."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        state = dc.apply_voltage(0.25)

    assert state.voltage == pytest.approx(0.25)
    assert state.is_on
    apply_call = next(
        call for call in dc_controller.calls if call[0] == "apply_voltage"
    )
    assert apply_call[1:3] == (7, 0.25)
    applied_profile = apply_call[3]
    assert isinstance(applied_profile, DCVoltageProfile)
    assert applied_profile.ramp_rate_v_per_s == pytest.approx(1.0)
    assert applied_profile.ramp_step_size_v == pytest.approx(0.1)


def test_dc_voltage_control_ramps_to_idle_voltage_on_exit() -> None:
    """Context exit should ramp back to the configured idle voltage."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        dc.apply_voltage(0.25)

    assert dc_controller.voltages[7] == pytest.approx(0.0)
    assert dc_controller.calls[-1][0:2] == ("idle", 7)


def test_dc_voltage_control_can_apply_voltage_immediately() -> None:
    """Explicit immediate application should skip intermediate ramp setpoints."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        state = dc.apply_voltage_immediately(0.2)

    assert any(call[0] == "apply_voltage_immediately" for call in dc_controller.calls)
    assert state.is_on


def test_dc_voltage_control_attempts_idle_even_when_the_ramp_fails() -> None:
    """A failing idle ramp on exit should still be attempted and surfaced."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    def exit_with_failed_restore() -> None:
        with ctx.dc_voltage_control() as dc:
            dc.apply_voltage(0.2)
            dc_controller.fail_set_voltage = True

    with pytest.raises(RuntimeError, match="restore failed"):
        exit_with_failed_restore()

    assert dc_controller.calls[-1][0:2] == ("idle", 7)


def test_dc_voltage_control_idles_after_partial_sweep() -> None:
    """Given a partial DC sweep, exiting its control should idle the output."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        states = dc.sweep(sweep_range=[0.1, 0.2])
        assert next(states).voltage == pytest.approx(0.1)

    assert dc_controller.voltages[7] == pytest.approx(0.0)


def test_dc_voltage_control_sets_and_reads_bound_mux() -> None:
    """Given a bound DC control, set and state should use the selected mux."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    with ctx.dc_voltage_control(mux=7) as dc:
        state = dc.apply_voltage(0.4)

        assert state == dc.state
        assert state.mux_label == "MUX07"
        assert state.channel == 8
        assert state.voltage == pytest.approx(0.4)
        assert state.is_on

    assert dc_controller.voltages[8] == pytest.approx(0.0)


def test_dc_voltage_control_uses_configured_readback_tolerance() -> None:
    """Voltage application should delegate the configured readback tolerance."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    with ctx.dc_voltage_control() as dc:
        state = dc.apply_voltage_immediately(0.5)

    assert state.voltage == pytest.approx(0.5)
    call = next(
        call for call in dc_controller.calls if call[0] == "apply_voltage_immediately"
    )
    applied_profile = call[3]
    assert isinstance(applied_profile, DCVoltageProfile)
    assert applied_profile.readback_tolerance_v == pytest.approx(0.002)


def test_get_dc_voltage_states_reads_all_wired_muxes_on_one_call() -> None:
    """Bulk state readback should cover every wired mux with one bulk read."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages.update({7: 0.12, 8: -0.34})
    dc_controller.output_states.update({7: True, 8: False})
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    states = ctx.get_dc_voltage_states()

    assert sorted(states) == [6, 7]
    assert states[6].mux_label == "MUX06"
    assert states[6].channel == 7
    assert states[6].voltage == pytest.approx(0.12)
    assert states[6].output == "on"
    assert states[7].channel == 8
    assert states[7].voltage == pytest.approx(-0.34)
    assert states[7].output == "off"
    assert dc_controller.calls == [("read_channels", (7, 8))]


def test_idle_dc_voltages_idles_all_wired_muxes() -> None:
    """Bulk idling should ramp every wired mux and return the new states."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages.update({7: 0.5, 8: 0.5})
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    states = ctx.idle_dc_voltages(confirm=False)

    assert dc_controller.calls[0] == ("idle_channels", (7, 8))
    assert states[6].voltage == pytest.approx(0.0)
    assert states[7].voltage == pytest.approx(0.0)


def test_bias_dc_voltages_applies_calibrated_amplification_points() -> None:
    """Bulk biasing should ramp each calibrated mux to its jpa_params voltage."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    states = ctx.bias_dc_voltages(confirm=False)

    assert dc_controller.calls[0] == ("apply_channels", (7, 8))
    assert states[6].voltage == pytest.approx(0.76)
    assert states[7].voltage == pytest.approx(0.42)


def test_bias_dc_voltages_skips_uncalibrated_muxes() -> None:
    """Muxes without a calibrated dc_voltage should not be biased."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06"],
        dc_controller=dc_controller,
        mux_to_channel={6: 1, 7: 4},
    )
    # mux 7 has no calibrated bias_voltage
    ctx.system_manager.experiment_system.control_params = _ControlParams({6: 0.76})

    states = ctx.bias_dc_voltages(confirm=False)

    assert dc_controller.calls[0] == ("apply_channels", (1,))
    assert states[6].voltage == pytest.approx(0.76)
    assert states[7].voltage == pytest.approx(0.0)


def test_bias_dc_voltages_can_be_cancelled_at_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the confirmation prompt should not write to the hardware."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)
    monkeypatch.setattr(
        "qubex.experiment.experiment_context.Confirm.ask",
        lambda *_args, **_kwargs: False,
    )

    states = ctx.bias_dc_voltages()

    assert all(call[0] != "apply_channels" for call in dc_controller.calls)
    assert states[6].voltage == pytest.approx(0.0)


def test_reset_dc_voltages_brings_every_wired_mux_to_initial_state() -> None:
    """Resetting should target every wired mux, on or off."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages.update({7: 0.5, 8: 15.0})
    dc_controller.output_states.update({7: True, 8: False})
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    states = ctx.reset_dc_voltages(confirm=False)

    init_calls = [call for call in dc_controller.calls if call[0] == "reset_channels"]
    assert init_calls == [("reset_channels", (7, 8))]
    assert states[6].voltage == pytest.approx(0.0)
    assert states[6].output == "on"
    assert states[7].voltage == pytest.approx(0.0)
    assert states[7].output == "on"


def test_shutdown_dc_voltages_turns_off_every_wired_mux() -> None:
    """Shutdown should ramp every wired mux to reset voltage and turn it off."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages.update({7: 0.5, 8: -0.4})
    dc_controller.output_states.update({7: True, 8: True})
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)

    states = ctx.shutdown_dc_voltages(confirm=False)

    assert ("turn_off_channels", (7, 8)) in dc_controller.calls
    assert states[6].voltage == pytest.approx(0.0)
    assert states[6].output == "off"
    assert states[7].voltage == pytest.approx(0.0)
    assert states[7].output == "off"


def test_shutdown_can_be_cancelled_at_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining shutdown confirmation should leave outputs untouched."""
    dc_controller = _DCVoltageController()
    dc_controller.output_states.update({7: True, 8: True})
    ctx = _ContextForTest(mux_labels=["MUX06"], dc_controller=dc_controller)
    monkeypatch.setattr(
        "qubex.experiment.experiment_context.Confirm.ask",
        lambda *_args, **_kwargs: False,
    )

    states = ctx.shutdown_dc_voltages()

    assert dc_controller.output_states[7] is True
    assert states[6].output == "on"
    assert all(call[0] != "turn_off_channels" for call in dc_controller.calls)


def test_bulk_operations_accept_a_mux_selection() -> None:
    """`muxes=` should narrow bias/idle/reset to the selected muxes."""
    dc_controller = _DCVoltageController()
    dc_controller.voltages.update({7: 0.5, 8: 0.5})
    dc_controller.output_states.update({7: True, 8: True})
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )

    ctx.bias_dc_voltages(muxes=6, confirm=False)
    ctx.idle_dc_voltages(muxes=["MUX07"], confirm=False)
    ctx.reset_dc_voltages(muxes=[6], confirm=False)
    ctx.shutdown_dc_voltages(muxes=["MUX07"], confirm=False)

    assert ("apply_channels", (7,)) in dc_controller.calls
    assert ("idle_channels", (8,)) in dc_controller.calls
    assert ("reset_channels", (7,)) in dc_controller.calls
    assert ("turn_off_channels", (8,)) in dc_controller.calls


def test_bias_raises_for_an_explicitly_selected_uncalibrated_mux() -> None:
    """Explicit selection of an uncalibrated mux should fail, not skip."""
    dc_controller = _DCVoltageController()
    ctx = _ContextForTest(
        mux_labels=["MUX06", "MUX07"],
        dc_controller=dc_controller,
    )
    ctx.system_manager.experiment_system.control_params = _ControlParams({6: 0.76})

    with pytest.raises(ValueError, match="no calibrated `bias_voltage`"):
        ctx.bias_dc_voltages(muxes=[6, 7], confirm=False)
