"""Tests for DC voltage backend helpers."""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import pytest

from qubex.external_devices.dc_voltage import (
    DCVoltageController,
    DCVoltageControllerConfig,
    DCVoltageProfile,
    ExternalDevicesConfig,
    create_dc_voltage_controller,
)
from qubex.external_devices.dc_voltage.drivers import ONS61797Device
from qubex.external_devices.dc_voltage.registry import DC_VOLTAGE_DRIVER_REGISTRY


class _FakeDCVoltageDevice:
    # Hardware state is class-level: it persists across connections, like the
    # real instrument.
    instances: ClassVar[list[_FakeDCVoltageDevice]] = []
    output_states: ClassVar[dict[int, bool]] = {}
    voltages: ClassVar[dict[int, float]] = {}
    calls: ClassVar[list[tuple[Any, ...]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.connect_kwargs: list[dict[str, Any]] = []
        self.closed = False
        _FakeDCVoltageDevice.instances.append(self)

    def connect(
        self,
        port: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        kwargs = {}
        if port is not None:
            kwargs["port"] = port
        if ip_address is not None:
            kwargs["ip_address"] = ip_address
        self.connect_kwargs.append(kwargs)

    @property
    def supports_native_ramp(self) -> bool:
        """Return that the fake device uses controller-generated ramps."""
        return False

    @property
    def supports_output_switch(self) -> bool:
        """Return that the fake device supports physical output switching."""
        return True

    def ramp_voltage(
        self,
        channel: int,
        start_voltage: float,
        target_voltage: float,
        rate_v_per_s: float,
        step_size_v: float,
        wait_s: float,
    ) -> None:
        """Reject native ramping for the generic fake device."""
        raise AssertionError(
            (channel, start_voltage, target_voltage, rate_v_per_s, step_size_v, wait_s)
        )

    def close(self) -> None:
        self.closed = True

    def on(self, channel: int) -> None:
        self.calls.append(("on", channel))
        self.output_states[channel] = True

    def off(self, channel: int) -> None:
        self.calls.append(("off", channel))
        self.output_states[channel] = False

    def set_voltage(self, channel: int, voltage: float) -> None:
        self.calls.append(("set_voltage", channel, voltage))
        self.voltages[channel] = voltage

    def get_voltage(self, channel: int) -> float:
        self.calls.append(("get_voltage", channel))
        return self.voltages[channel]

    def is_output_on(self, channel: int) -> bool:
        return self.output_states.get(channel, False)


class _FakeONS61797Client:
    def __init__(self, **_: Any) -> None:
        self.output_states: dict[int, int] = {}
        self.voltages: dict[int, float] = {}

    def on(self, channel: int) -> None:
        self.output_states[channel] = 1

    def get_output_state(self, channel: int) -> int:
        return self.output_states.get(channel, 0)

    def get_voltage(self, *, channel: int) -> float:
        return self.voltages.get(channel, 0.0)


def _reset_fake_devices() -> None:
    _FakeDCVoltageDevice.instances = []
    _FakeDCVoltageDevice.output_states = {}
    _FakeDCVoltageDevice.voltages = {1: 0.1, 2: -0.2}
    _FakeDCVoltageDevice.calls = []


def test_factory_resolves_registered_driver_with_opaque_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered driver should receive its unchanged connection mapping."""
    _reset_fake_devices()
    connections: list[tuple[str, dict[str, object], tuple[int, ...] | None]] = []

    def build_device_factory(
        device_id: str,
        connection: dict[str, object],
        ports: tuple[int, ...] | None,
    ):
        connections.append((device_id, connection, ports))
        return partial(_FakeDCVoltageDevice, resource=connection["resource"])

    monkeypatch.setitem(
        DC_VOLTAGE_DRIVER_REGISTRY,
        "fake-dc",
        build_device_factory,
    )
    config = ExternalDevicesConfig.from_dict(
        {
            "devices": {
                "FAKE1": {
                    "driver": "fake-dc",
                    "params": {"resource": "external-a"},
                    "channels": [1],
                },
            },
            "wiring": [{"mux": 0, "bias": "FAKE1-1"}],
        }
    ).dc_voltage

    controller = create_dc_voltage_controller(config)
    controller.read_channels([1])

    assert connections == [("FAKE1", {"resource": "external-a"}, (1,))]
    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"resource": "external-a"}


def test_factory_creates_controller_from_configured_serial_port() -> None:
    """Given controller config, factory should create a configured controller."""
    _reset_fake_devices()
    config = DCVoltageControllerConfig(
        driver="ons61797",
        params={"port": "/dev/system-dc"},
        device_factory=_FakeDCVoltageDevice,
    )

    controller = create_dc_voltage_controller(config)
    controller.read_channels([1])

    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {}


def test_ons61797_driver_validates_its_own_connection_options() -> None:
    """ONS61797 should reject conflicting transport settings in its driver layer."""
    _reset_fake_devices()
    config = ExternalDevicesConfig.from_dict(
        {
            "devices": {
                "ONS1": {
                    "driver": "ons61797",
                    "params": {
                        "port": "/dev/serial-dc",
                        "ip_address": "192.0.2.20",
                    },
                },
            },
            "wiring": [{"mux": 0, "bias": "ONS1-1"}],
        }
    ).dc_voltage

    with pytest.raises(TypeError, match="Only one"):
        create_dc_voltage_controller(config)


def test_factory_rejects_unknown_driver() -> None:
    """Given unknown driver, factory should reject it before hardware access."""
    config = DCVoltageControllerConfig(driver="unknown")

    with pytest.raises(ValueError, match="Unsupported DC voltage controller driver"):
        create_dc_voltage_controller(config)


def test_external_devices_config_resolves_device_output_refs() -> None:
    """Mux outputs should resolve `DEVICE-CHANNEL` refs against `devices`."""
    config = ExternalDevicesConfig.from_dict(
        {
            "devices": {
                "ONS1": {
                    "driver": "ons61797",
                    "params": {"port": "/dev/ttyACM0"},
                    "channels": [9, 10],
                },
            },
            "wiring": [
                {"mux": 8, "bias": "ONS1-9"},
                {"mux": 9, "bias": "ONS1-10"},
            ],
            "settings": {
                "ramp": {"rate_v_per_s": 0.2},
                "overrides": [
                    {"mux": 9, "ramp": {"rate_v_per_s": 0.05}},
                ],
            },
        }
    )

    controller = config.dc_voltage
    assert controller.driver == "ons61797"
    assert controller.params == {"port": "/dev/ttyACM0"}
    assert controller.device_id == "ONS1"
    assert controller.channels == (9, 10)
    assert controller.resolve_voltage_profile(8).channel == 9
    assert controller.resolve_voltage_profile(8).ramp_rate_v_per_s == 0.2
    assert controller.resolve_voltage_profile(9).channel == 10
    assert controller.resolve_voltage_profile(9).ramp_rate_v_per_s == 0.05


def test_settings_reset_voltage_resolves_with_overrides() -> None:
    """`reset_voltage` should default from settings and override per mux."""
    config = ExternalDevicesConfig.from_dict(
        {
            "devices": {"ONS1": {"driver": "ons61797"}},
            "wiring": [
                {"mux": 0, "bias": "ONS1-1"},
                {"mux": 1, "bias": "ONS1-2"},
            ],
            "settings": {
                "reset_voltage": -0.1,
                "overrides": [{"mux": 1, "reset_voltage": 0.05}],
            },
        }
    )

    controller = config.dc_voltage
    assert controller.resolve_voltage_profile(0).reset_voltage_v == -0.1
    assert controller.resolve_voltage_profile(1).reset_voltage_v == 0.05
    # Without calibration, the idle voltage falls back to the reset voltage.
    assert controller.resolve_voltage_profile(0).idle_voltage_v == -0.1
    assert controller.resolve_voltage_profile(1).idle_voltage_v == 0.05


def test_settings_no_longer_accept_idle_voltage() -> None:
    """The idle voltage is calibration (`jpa_params.yaml`), not settings."""
    with pytest.raises(ValueError, match="Unknown DC voltage settings"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {"ONS1": {"driver": "ons61797"}},
                "wiring": [{"mux": 0, "bias": "ONS1-1"}],
                "settings": {"idle_voltage_v": 0.1},
            }
        )


def test_settings_reject_unknown_keys() -> None:
    """Unknown settings and override keys should fail at parse time."""
    base = {
        "devices": {"ONS1": {"driver": "ons61797"}},
        "wiring": [{"mux": 0, "bias": "ONS1-1"}],
    }
    with pytest.raises(ValueError, match="Unknown DC voltage settings"):
        ExternalDevicesConfig.from_dict({**base, "settings": {"exit_mode": "idle"}})
    with pytest.raises(ValueError, match="Unknown DC voltage settings"):
        ExternalDevicesConfig.from_dict({**base, "settings": {"reset_voltage_v": 0.0}})
    with pytest.raises(ValueError, match="Unknown `overrides` settings"):
        ExternalDevicesConfig.from_dict(
            {**base, "settings": {"overrides": [{"mux": 0, "shutdown": {}}]}}
        )


def test_mux_output_rejects_unknown_device() -> None:
    """An output referencing an undefined device should fail at parse time."""
    with pytest.raises(ValueError, match="unknown device 'QBLOX1'"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {
                    "ONS1": {"driver": "ons61797"},
                },
                "wiring": [{"mux": 8, "bias": "QBLOX1-15"}],
            }
        )


def test_mux_outputs_must_reference_one_device() -> None:
    """One controller should reject outputs spread across devices."""
    with pytest.raises(ValueError, match="same device"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {
                    "ONS1": {"driver": "ons61797"},
                    "ONS2": {"driver": "ons61797"},
                },
                "wiring": [
                    {"mux": 0, "bias": "ONS1-1"},
                    {"mux": 1, "bias": "ONS2-2"},
                ],
            }
        )


def test_mux_output_channel_must_be_whitelisted() -> None:
    """An output channel outside the device `channels` list should fail."""
    with pytest.raises(ValueError, match="not in device 'ONS1'"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {
                    "ONS1": {"driver": "ons61797", "channels": [1, 2]},
                },
                "wiring": [{"mux": 0, "bias": "ONS1-3"}],
            }
        )


@pytest.mark.parametrize("output", ["ONS1", "ONS1-", "-15", "ONS1-x", 15, None])
def test_mux_output_rejects_malformed_refs(output: object) -> None:
    """Outputs must use the `DEVICE-CHANNEL` reference form."""
    with pytest.raises((TypeError, ValueError), match="DEVICE-CHANNEL"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {"ONS1": {"driver": "ons61797"}},
                "wiring": [{"mux": 0, "bias": output}],
            }
        )


def test_top_level_and_wiring_reject_unknown_keys() -> None:
    """Unknown top-level sections and malformed wiring values should fail."""
    with pytest.raises(ValueError, match="Unknown external-device settings"):
        ExternalDevicesConfig.from_dict({"dc_voltage_controllers": {}})
    with pytest.raises(ValueError, match="Unknown DC voltage settings"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {"ONS1": {"driver": "ons61797"}},
                "wiring": [{"mux": 0, "bias": "ONS1-1"}],
                "settings": {"driver": "ons61797"},
            }
        )
    with pytest.raises(TypeError, match="DEVICE-CHANNEL"):
        ExternalDevicesConfig.from_dict(
            {
                "devices": {"ONS1": {"driver": "ons61797"}},
                "wiring": [{"mux": 0, "channel": 1}],
            }
        )


def test_apply_voltages_ramps_each_channel_and_returns_to_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporary voltage application should ramp up and back to idle."""
    _reset_fake_devices()
    delays: list[float] = []
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        delays.append,
    )
    _FakeDCVoltageDevice.output_states = {1: True}
    _FakeDCVoltageDevice.voltages = {1: 0.0}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
    )

    with controller.apply_voltages({1: (0.25, profile)}):
        pass

    applied = [
        call[2] for call in _FakeDCVoltageDevice.calls if call[0] == "set_voltage"
    ]
    assert applied == pytest.approx([0.1, 0.2, 0.25, 0.15, 0.05, 0.0])
    assert all(call[0] != "off" for call in _FakeDCVoltageDevice.calls)
    assert delays == [0.1] * 6
    assert len(_FakeDCVoltageDevice.instances) == 2
    assert all(device.closed for device in _FakeDCVoltageDevice.instances)


def test_apply_voltage_ramps_with_one_device_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-channel application should keep one device open for the ramp."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True}
    _FakeDCVoltageDevice.voltages = {1: 0.0}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
    )

    controller.apply_voltage(channel=1, voltage=0.25, profile=profile)

    assert len(_FakeDCVoltageDevice.instances) == 1
    device = _FakeDCVoltageDevice.instances[0]
    applied = [call[2] for call in device.calls if call[0] == "set_voltage"]
    assert applied == pytest.approx([0.1, 0.2, 0.25])
    readbacks = [call for call in device.calls if call[0] == "get_voltage"]
    assert readbacks == [("get_voltage", 1), ("get_voltage", 1)]
    assert device.output_states[1] is True
    assert device.closed is True


def test_idle_ramps_to_idle_voltage_without_touching_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idling should ramp to the idle voltage and leave the output switch alone."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )

    class _EnabledDevice(_FakeDCVoltageDevice):
        def __init__(self) -> None:
            super().__init__()
            self.output_states[1] = True
            self.voltages[1] = 0.25

    controller = DCVoltageController(device_factory=_EnabledDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
    )

    controller.idle(channel=1, profile=profile)

    device = _FakeDCVoltageDevice.instances[0]
    applied = [call[2] for call in device.calls if call[0] == "set_voltage"]
    assert applied == pytest.approx([0.15, 0.05, 0.0])
    assert all(call[0] != "off" for call in device.calls)


def test_apply_channels_ramps_each_channel_on_one_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk application should ramp every channel with a single connection."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True, 2: True}
    _FakeDCVoltageDevice.voltages = {1: 0.0, 2: 0.0}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
    )

    controller.apply_channels({1: (0.2, profile), 2: (-0.1, profile)})

    assert _FakeDCVoltageDevice.voltages[1] == pytest.approx(0.2)
    assert _FakeDCVoltageDevice.voltages[2] == pytest.approx(-0.1)
    assert len(_FakeDCVoltageDevice.instances) == 1
    assert _FakeDCVoltageDevice.instances[0].closed is True


def test_reset_channels_brings_all_channels_to_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resetting should ramp on channels and switch on off channels."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True, 2: False}
    _FakeDCVoltageDevice.voltages = {1: 0.2, 2: 15.0}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        reset_voltage_v=0.0,
    )

    controller.reset_channels({1: profile, 2: profile})

    assert _FakeDCVoltageDevice.voltages[1] == pytest.approx(0.0)  # ramped
    assert _FakeDCVoltageDevice.voltages[2] == pytest.approx(0.0)  # overwritten
    assert _FakeDCVoltageDevice.output_states == {1: True, 2: True}
    assert len(_FakeDCVoltageDevice.instances) == 1


def test_turn_off_channels_ramps_to_zero_before_switching_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk turn-off should leave every output off with a reset setpoint."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True, 2: False}
    _FakeDCVoltageDevice.voltages = {1: 0.2, 2: 0.5}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
    )

    controller.turn_off_channels({1: profile, 2: profile})

    off_calls = [call for call in _FakeDCVoltageDevice.calls if call[0] == "off"]
    assert off_calls == [("off", 1)]
    assert _FakeDCVoltageDevice.voltages[1] == pytest.approx(0.0)
    assert _FakeDCVoltageDevice.voltages[2] == pytest.approx(0.0)
    assert _FakeDCVoltageDevice.output_states == {1: False, 2: False}


def test_read_channels_reads_all_channels_on_one_connection() -> None:
    """Bulk readback should read every channel with a single connection."""
    _reset_fake_devices()
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)

    readings = controller.read_channels([1, 2])

    assert readings == {1: (0.1, False), 2: (-0.2, False)}
    assert len(_FakeDCVoltageDevice.instances) == 1


def test_turn_off_channels_skips_switch_for_devices_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown should stop after ramping when output switching is unavailable."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True}
    _FakeDCVoltageDevice.voltages = {1: 0.2}

    class _NoSwitchDevice(_FakeDCVoltageDevice):
        @property
        def supports_output_switch(self) -> bool:
            return False

        def off(self, channel: int) -> None:
            raise AssertionError(channel)

    controller = DCVoltageController(device_factory=_NoSwitchDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        reset_voltage_v=0.0,
    )

    controller.turn_off_channels({1: profile})

    assert _FakeDCVoltageDevice.voltages[1] == pytest.approx(0.0)
    assert all(call[0] != "off" for call in _FakeDCVoltageDevice.calls)
    assert _FakeDCVoltageDevice.instances[0].closed is True


def test_idle_channels_ramps_each_channel_on_one_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk idling should ramp every channel with a single connection."""
    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.voltages = {1: 0.2, 2: -0.1}
    _FakeDCVoltageDevice.output_states = {1: True, 2: True}
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
    )

    controller.idle_channels({1: profile, 2: profile})

    assert _FakeDCVoltageDevice.voltages[1] == pytest.approx(0.0)
    assert _FakeDCVoltageDevice.voltages[2] == pytest.approx(0.0)
    assert len(_FakeDCVoltageDevice.instances) == 1
    assert _FakeDCVoltageDevice.instances[0].closed is True


def test_apply_voltages_retries_until_readback_is_within_profile_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporary voltage application should verify setpoints with configured retries."""

    class _DelayedReadbackDevice(_FakeDCVoltageDevice):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.set_attempts = 0

        def set_voltage(self, channel: int, voltage: float) -> None:
            super().set_voltage(channel, voltage)
            self.set_attempts += 1

        def get_voltage(self, channel: int) -> float:
            self.calls.append(("get_voltage", channel))
            if self.set_attempts == 1:
                return self.voltages[channel] + 0.01
            return self.voltages[channel]

    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    _FakeDCVoltageDevice.output_states = {1: True}
    _FakeDCVoltageDevice.voltages = {1: 0.0}
    controller = DCVoltageController(
        device_factory=_DelayedReadbackDevice,
    )
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        ramp_step_size_v=0.1,
        ramp_wait_s=0.1,
        idle_voltage_v=0.0,
        readback_tolerance_v=0.001,
        max_set_attempts=2,
    )

    with controller.apply_voltages({1: (0.1, profile)}):
        pass

    device = _FakeDCVoltageDevice.instances[0]
    first_set = next(
        index for index, call in enumerate(device.calls) if call[0] == "set_voltage"
    )
    assert device.calls[first_set : first_set + 4] == [
        ("set_voltage", 1, 0.1),
        ("get_voltage", 1),
        ("set_voltage", 1, 0.1),
        ("get_voltage", 1),
    ]


def test_apply_voltage_stops_after_configured_readback_attempts() -> None:
    """Single-channel application should fail after its profile retry limit."""

    class _MismatchedReadbackDevice(_FakeDCVoltageDevice):
        read_count = 0

        def get_voltage(self, channel: int) -> float:
            self.calls.append(("get_voltage", channel))
            self.read_count += 1
            if self.read_count == 1:
                return 0.0
            return -1.0

    _reset_fake_devices()
    _FakeDCVoltageDevice.output_states = {1: True}
    controller = DCVoltageController(device_factory=_MismatchedReadbackDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_step_size_v=1.0,
        readback_tolerance_v=0.001,
        max_set_attempts=2,
    )

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        controller.apply_voltage(
            channel=1,
            voltage=0.5,
            profile=profile,
        )

    device = _FakeDCVoltageDevice.instances[0]
    set_calls = [call for call in device.calls if call[0] == "set_voltage"]
    assert len(set_calls) == 2


def test_apply_voltage_requires_the_output_to_be_on() -> None:
    """Applying to an off output should fail instead of switching it on."""
    _reset_fake_devices()
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(channel=1)

    with pytest.raises(RuntimeError, match="reset_dc_voltages"):
        controller.apply_voltage(channel=1, voltage=0.1, profile=profile)
    assert all(call[0] != "on" for call in _FakeDCVoltageDevice.calls)


def test_ons61797_turn_on_refuses_out_of_range_stored_setpoint() -> None:
    """Turning on should fail while the stored setpoint is outside the range."""
    from qubex.external_devices.dc_voltage.drivers import ONS61797Device

    client = _FakeONS61797Client()
    device = ONS61797Device(
        port="/dev/fake",
        client_factory=lambda **_: client,
    )
    client.voltages[1] = 15.0

    with pytest.raises(ValueError, match="outside"):
        device.on(1)
    assert client.get_output_state(channel=1) == 0

    client.voltages[1] = 0.0
    device.on(1)
    assert client.get_output_state(channel=1) == 1


def test_ons61797_rejects_voltages_outside_the_allowed_range() -> None:
    """Writes outside -4 V to 4 V should fail before reaching hardware."""
    from qubex.external_devices.dc_voltage.drivers import ONS61797Device

    written: list[tuple[int, float]] = []

    class _Client:
        def __init__(self, **_: object) -> None:
            pass

        def set_voltage(self, *, channel: int, voltage: float) -> None:
            written.append((channel, voltage))

    device = ONS61797Device(port="/dev/fake", client_factory=_Client)
    with pytest.raises(ValueError, match=r"between -4\.0 V and 4\.0 V"):
        device.set_voltage(1, 15.0)
    assert written == []

    device.set_voltage(1, 3.5)
    assert written == [(1, 3.5)]


def test_ons61797_adapter_normalizes_third_party_output_state() -> None:
    """Given a third-party client, ONS61797 adapter should expose a boolean state."""
    device = ONS61797Device(
        port="/dev/test-dc",
        client_factory=_FakeONS61797Client,
    )

    device.on(1)

    assert device.is_output_on(1) is True
