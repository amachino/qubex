"""DC voltage driver registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .config import DCVoltageProfile
from .drivers.ons61797 import (
    create_ons61797_device_factory,
    validate_ons61797_profile,
    validate_ons61797_voltage,
)
from .drivers.qblox_backend import (
    create_qblox_backend_device_factory,
    validate_qblox_backend_profile,
    validate_qblox_backend_voltage,
)
from .protocol import DCVoltageDeviceFactory

DCVoltageDriverFactory = Callable[
    [str, Mapping[str, object], Sequence[int] | None],
    DCVoltageDeviceFactory,
]


@dataclass(frozen=True)
class DCVoltageDriverSpec:
    """Register one device factory and its optional safety validators."""

    create_device_factory: DCVoltageDriverFactory
    validate_profile: Callable[[DCVoltageProfile], None] | None = None
    validate_voltage: Callable[[float], None] | None = None


DC_VOLTAGE_DRIVER_REGISTRY: dict[str, DCVoltageDriverSpec] = {
    "ons61797": DCVoltageDriverSpec(
        create_device_factory=create_ons61797_device_factory,
        validate_profile=validate_ons61797_profile,
        validate_voltage=validate_ons61797_voltage,
    ),
    "qblox_backend": DCVoltageDriverSpec(
        create_device_factory=create_qblox_backend_device_factory,
        validate_profile=validate_qblox_backend_profile,
        validate_voltage=validate_qblox_backend_voltage,
    ),
}
