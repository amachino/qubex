"""Tests for DC voltage backend helpers."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from qubex.backend.dc_voltage import (
    DCVoltageController,
    DCVoltageControllerConfig,
    create_dc_voltage_controller,
)
from qubex.backend.dc_voltage.drivers import ONS61797Device


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


def test_controller_uses_injected_connection_options_for_each_call() -> None:
    """Given custom connection options, controller operations should use them."""
    _reset_fake_devices()
    controller = DCVoltageController(
        port="/dev/custom-dc",
        device_factory=_FakeDCVoltageDevice,
    )

    controller.set_voltage(1, 0.7)
    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"port": "/dev/custom-dc"}
    assert _FakeDCVoltageDevice.instances[0].closed is True

    controller.get_voltage(1)
    assert _FakeDCVoltageDevice.instances[0].connect_kwargs == [
        {"port": "/dev/custom-dc"}
    ]


def test_controller_accepts_ip_address_instead_of_serial_port() -> None:
    """Given IP connection options, controller should not fall back to serial port."""
    _reset_fake_devices()
    controller = DCVoltageController(
        port=None,
        ip_address="192.0.2.10",
        device_factory=_FakeDCVoltageDevice,
    )

    controller.on(1)

    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"ip_address": "192.0.2.10"}


def test_factory_creates_controller_from_configured_serial_port() -> None:
    """Given controller config, factory should create a configured controller."""
    _reset_fake_devices()
    config = DCVoltageControllerConfig(
        driver="ons61797",
        port="/dev/system-dc",
        device_factory=_FakeDCVoltageDevice,
    )

    controller = create_dc_voltage_controller(config)
    controller.on(1)

    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"port": "/dev/system-dc"}


def test_controllers_keep_independent_connection_configuration() -> None:
    """Given two controllers, each should retain its own connection configuration."""
    _reset_fake_devices()
    serial = DCVoltageController(
        port="/dev/serial-dc",
        device_factory=_FakeDCVoltageDevice,
    )
    network = DCVoltageController(
        port=None,
        ip_address="192.0.2.20",
        device_factory=_FakeDCVoltageDevice,
    )

    serial.on(1)
    network.on(2)

    assert serial is not network
    assert [device.init_kwargs for device in _FakeDCVoltageDevice.instances] == [
        {"port": "/dev/serial-dc"},
        {"ip_address": "192.0.2.20"},
    ]


def test_factory_rejects_unknown_driver() -> None:
    """Given unknown driver, factory should reject it before hardware access."""
    config = DCVoltageControllerConfig(driver="unknown")

    with pytest.raises(ValueError, match="Unsupported DC voltage controller driver"):
        create_dc_voltage_controller(config)


def test_ons61797_adapter_normalizes_third_party_output_state() -> None:
    """Given a third-party client, ONS61797 adapter should expose a boolean state."""
    device = ONS61797Device(
        port="/dev/test-dc",
        client_factory=_FakeONS61797Client,
    )

    device.on(1)

    assert device.is_output_on(1) is True
