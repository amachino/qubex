"""DC voltage controller configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import DCVoltageDeviceFactory


@dataclass(frozen=True)
class DCVoltageControllerConfig:
    """Configure a backend-managed DC voltage controller."""

    driver: str = "ons61797"
    port: str | None = None
    ip_address: str | None = None
    device_factory: DCVoltageDeviceFactory | None = None
    mux_to_channel: dict[int, int] = field(default_factory=dict)
