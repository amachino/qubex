"""Aggregate configuration for external-device categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .dc_voltage.config import DCVoltageConfig


@runtime_checkable
class ExternalDeviceConfigSection(Protocol):
    """Define the construction contract for one external-device category."""

    @classmethod
    def from_dict(cls, raw_config: object) -> ExternalDeviceConfigSection:
        """Create one device-category section from external-device data."""
        ...


@dataclass(frozen=True)
class ExternalDevicesConfig:
    """Compose all configured external-device categories."""

    dc_voltage: DCVoltageConfig = field(default_factory=DCVoltageConfig)

    @classmethod
    def from_dict(cls, raw_config: object) -> ExternalDevicesConfig:
        """Create aggregate external-device configuration from one mapping."""
        return cls(dc_voltage=DCVoltageConfig.from_dict(raw_config))
