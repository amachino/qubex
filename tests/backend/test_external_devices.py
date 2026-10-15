"""Tests for aggregate external-device configuration and lifecycle ownership."""

from qubex.external_devices import (
    DCVoltageConfig,
    ExternalDeviceConfigSection,
    ExternalDevicesConfig,
    ExternalDevicesController,
)


def test_external_devices_config_composes_typed_device_sections() -> None:
    """The aggregate config should expose a typed DC-voltage section."""
    config = ExternalDevicesConfig.from_dict(None)

    assert isinstance(config.dc_voltage, DCVoltageConfig)
    assert isinstance(config.dc_voltage, ExternalDeviceConfigSection)


def test_external_devices_controller_owns_typed_controllers() -> None:
    """The lifecycle container should expose its typed DC-voltage controller."""
    config = ExternalDevicesConfig()

    controller = ExternalDevicesController(config)

    assert controller.config is config
    assert controller.dc_voltage is not None
