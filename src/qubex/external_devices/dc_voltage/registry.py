"""DC voltage driver registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .drivers.ons61797 import create_ons61797_device_factory
from .drivers.qblox_server import create_qblox_server_device_factory
from .protocol import DCVoltageDeviceFactory

DCVoltageDriverFactory = Callable[
    [Mapping[str, object]],
    DCVoltageDeviceFactory,
]

DC_VOLTAGE_DRIVER_REGISTRY: dict[str, DCVoltageDriverFactory] = {
    "ons61797": create_ons61797_device_factory,
    "qblox_server": create_qblox_server_device_factory,
}
