"""DC voltage driver registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .config import DCVoltageProfile
from .drivers.ons61797 import create_ons61797_device_factory
from .drivers.qblox_server import (
    create_qblox_server_device_factory,
    validate_qblox_server_profile,
)
from .protocol import DCVoltageDeviceFactory

DCVoltageDriverFactory = Callable[
    [str, Mapping[str, object], Sequence[int] | None],
    DCVoltageDeviceFactory,
]


@dataclass(frozen=True)
class DCVoltageDriverSpec:
    """Register one device factory and its optional profile validator."""

    create_device_factory: DCVoltageDriverFactory
    validate_profile: Callable[[DCVoltageProfile], None] | None = None


DC_VOLTAGE_DRIVER_REGISTRY: dict[str, DCVoltageDriverSpec] = {
    "ons61797": DCVoltageDriverSpec(
        create_device_factory=create_ons61797_device_factory,
    ),
    "qblox_server": DCVoltageDriverSpec(
        create_device_factory=create_qblox_server_device_factory,
        validate_profile=validate_qblox_server_profile,
    ),
}
