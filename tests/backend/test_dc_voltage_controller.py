"""Tests for DC voltage backend helpers."""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import pytest

from qubex.external_devices.dc_voltage import (
    DCVoltageController,
    DCVoltageControllerConfig,
    DCVoltageProfile,
    create_dc_voltage_controller,
)
from qubex.external_devices.dc_voltage.drivers import ONS61797Device
from qubex.external_devices.dc_voltage.registry import DC_VOLTAGE_DRIVER_REGISTRY


class _FakeDCVoltageDevice:
    instances: ClassVar[list[_FakeDCVoltageDevice]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.connect_kwargs: list[dict[str, Any]] = []
        self.closed = False
        self.output_states: dict[int, bool] = {}
        self.voltages = {1: 0.1, 2: -0.2}
        self.calls: list[tuple[Any, ...]] = []
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

    def on(self, channel: int) -> None:
        self.output_states[channel] = 1

    def get_output_state(self, channel: int) -> int:
        return self.output_states.get(channel, 0)


def _reset_fake_devices() -> None:
    _FakeDCVoltageDevice.instances = []


def test_controller_creates_and_closes_one_device_for_each_operation() -> None:
    """Each controller operation should use one short-lived device instance."""
    _reset_fake_devices()
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)

    controller.set_voltage(1, 0.7)
    assert _FakeDCVoltageDevice.instances[0].closed is True

    controller.get_voltage(1)
    assert len(_FakeDCVoltageDevice.instances) == 2
    assert all(device.closed for device in _FakeDCVoltageDevice.instances)


def test_factory_resolves_registered_driver_with_opaque_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered driver should receive its unchanged connection mapping."""
    _reset_fake_devices()
    connections: list[dict[str, object]] = []

    def build_device_factory(
        connection: dict[str, object],
    ):
        connections.append(connection)
        return partial(_FakeDCVoltageDevice, resource=connection["resource"])

    monkeypatch.setitem(
        DC_VOLTAGE_DRIVER_REGISTRY,
        "fake-dc",
        build_device_factory,
    )
    config = DCVoltageControllerConfig.from_dict(
        {
            "driver": "fake-dc",
            "connection": {"resource": "external-a"},
        }
    )

    controller = create_dc_voltage_controller(config)
    controller.on(1)

    assert connections == [{"resource": "external-a"}]
    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"resource": "external-a"}


def test_factory_creates_controller_from_configured_serial_port() -> None:
    """Given controller config, factory should create a configured controller."""
    _reset_fake_devices()
    config = DCVoltageControllerConfig(
        driver="ons61797",
        connection={"port": "/dev/system-dc"},
        device_factory=_FakeDCVoltageDevice,
    )

    controller = create_dc_voltage_controller(config)
    controller.on(1)

    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {}


def test_ons61797_driver_validates_its_own_connection_options() -> None:
    """ONS61797 should reject conflicting transport settings in its driver layer."""
    _reset_fake_devices()
    config = DCVoltageControllerConfig.from_dict(
        {
            "driver": "ons61797",
            "connection": {
                "port": "/dev/serial-dc",
                "ip_address": "192.0.2.20",
            },
        }
    )

    with pytest.raises(TypeError, match="Only one"):
        create_dc_voltage_controller(config)


def test_factory_rejects_unknown_driver() -> None:
    """Given unknown driver, factory should reject it before hardware access."""
    config = DCVoltageControllerConfig(driver="unknown")

    with pytest.raises(ValueError, match="Unsupported DC voltage controller driver"):
        create_dc_voltage_controller(config)


def test_apply_voltages_ramps_each_channel_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporary voltage application should ramp up and down before output off."""
    _reset_fake_devices()
    delays: list[float] = []
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        delays.append,
    )
    controller = DCVoltageController(device_factory=_FakeDCVoltageDevice)
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        update_interval_s=0.1,
        safe_voltage_v=0.0,
    )

    with controller.apply_voltages({1: (0.25, profile)}):
        pass

    device = _FakeDCVoltageDevice.instances[0]
    applied = [call[2] for call in device.calls if call[0] == "set_voltage"]
    assert applied == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.15, 0.05, 0.0])
    assert device.calls[-1] == ("off", 1)
    assert delays == [0.1] * 6


def test_apply_voltages_retries_until_readback_is_within_profile_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporary voltage application should verify setpoints with configured retries."""

    class _DelayedReadbackDevice(_FakeDCVoltageDevice):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.readback_attempts = 0

        def get_voltage(self, channel: int) -> float:
            self.calls.append(("get_voltage", channel))
            self.readback_attempts += 1
            if self.readback_attempts == 1:
                return self.voltages[channel] + 0.01
            return self.voltages[channel]

    _reset_fake_devices()
    monkeypatch.setattr(
        "qubex.external_devices.dc_voltage.controller.time.sleep",
        lambda _: None,
    )
    controller = DCVoltageController(
        device_factory=_DelayedReadbackDevice,
    )
    profile = DCVoltageProfile(
        channel=1,
        ramp_rate_v_per_s=1.0,
        update_interval_s=0.1,
        safe_voltage_v=0.0,
        readback_tolerance_v=0.001,
        max_set_attempts=2,
    )

    with controller.apply_voltages({1: (0.1, profile)}):
        pass

    device = _FakeDCVoltageDevice.instances[0]
    assert device.calls[:5] == [
        ("set_voltage", 1, 0.0),
        ("get_voltage", 1),
        ("set_voltage", 1, 0.0),
        ("get_voltage", 1),
        ("on", 1),
    ]


def test_ons61797_adapter_normalizes_third_party_output_state() -> None:
    """Given a third-party client, ONS61797 adapter should expose a boolean state."""
    device = ONS61797Device(
        port="/dev/test-dc",
        client_factory=_FakeONS61797Client,
    )

    device.on(1)

    assert device.is_output_on(1) is True
