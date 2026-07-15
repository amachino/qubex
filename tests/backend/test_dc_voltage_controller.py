"""Tests for DC voltage backend helpers."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest

from qubex.backend.dc_voltage_controller import (
    DCVoltageController,
    DCVoltageControllerConfig,
    create_dc_voltage_controller,
    dc_voltage,
)


class _FakeDCVoltageDevice:
    instances: ClassVar[list[_FakeDCVoltageDevice]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.connect_kwargs: list[dict[str, Any]] = []
        self.closed = False
        self.output_states: dict[int, int] = {}
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
        self.output_states[channel] = 1

    def off(self, channel: int) -> None:
        self.calls.append(("off", channel))
        self.output_states[channel] = 0

    def set_voltage(self, channel: int, voltage: float) -> None:
        self.calls.append(("set_voltage", channel, voltage))
        self.voltages[channel] = voltage

    def get_voltage(self, channel: int) -> float:
        self.calls.append(("get_voltage", channel))
        return self.voltages[channel]

    def get_output_state(self, channel: int) -> int:
        return self.output_states.get(channel, 0)

    def get_device_information(self) -> str:
        return "fake"

    def reset(self) -> None:
        self.calls.append(("reset",))


def _reset_fake_devices() -> None:
    _FakeDCVoltageDevice.instances = []


def test_dc_voltage_uses_injected_device_and_restores_original_values() -> None:
    """Given injected device, when applying voltages, then originals are restored."""
    _reset_fake_devices()

    with dc_voltage(
        {1: 0.5, 2: 0.6},
        port="/dev/test-dc",
        device_factory=_FakeDCVoltageDevice,
    ) as raw_device:
        device = cast(_FakeDCVoltageDevice, raw_device)
        assert device.init_kwargs == {"port": "/dev/test-dc"}
        assert device.voltages == {1: 0.5, 2: 0.6}
        assert device.output_states == {1: 1, 2: 1}

    assert device.voltages == {1: 0.1, 2: -0.2}
    assert device.output_states == {1: 0, 2: 0}
    assert device.closed is True


def test_controller_uses_injected_connection_options_for_each_call() -> None:
    """Given custom connection options, controller operations should use them."""
    _reset_fake_devices()
    DCVoltageController.reset_shared()
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
    DCVoltageController.reset_shared()
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
    DCVoltageController.reset_shared()
    config = DCVoltageControllerConfig(
        driver="ons61797",
        port="/dev/system-dc",
        device_factory=_FakeDCVoltageDevice,
    )

    controller = create_dc_voltage_controller(config)
    controller.on(1)

    assert _FakeDCVoltageDevice.instances[0].init_kwargs == {"port": "/dev/system-dc"}


def test_factory_rejects_unknown_driver() -> None:
    """Given unknown driver, factory should reject it before hardware access."""
    config = DCVoltageControllerConfig(driver="unknown")

    with pytest.raises(ValueError, match="Unsupported DC voltage controller driver"):
        create_dc_voltage_controller(config)
