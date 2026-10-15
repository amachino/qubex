"""Lifecycle container for typed external-device controllers."""

from __future__ import annotations

from .config import ExternalDevicesConfig
from .dc_voltage.controller import DCVoltageController, create_dc_voltage_controller


class ExternalDevicesController:
    """Own typed controllers for all configured external-device categories."""

    def __init__(self, config: ExternalDevicesConfig | None = None) -> None:
        """Create typed controllers from aggregate configuration."""
        self._config = config or ExternalDevicesConfig()
        self._dc_voltage = create_dc_voltage_controller(
            self._config.dc_voltage.controller
        )

    @property
    def config(self) -> ExternalDevicesConfig:
        """Return the aggregate configuration used by this container."""
        return self._config

    @property
    def dc_voltage(self) -> DCVoltageController:
        """Return the typed DC-voltage controller."""
        return self._dc_voltage
