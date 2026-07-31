"""External DC voltage controller."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from .config import DCVoltageControllerConfig, DCVoltageProfile
from .drivers import ONS61797Device
from .protocol import DCVoltageDevice, DCVoltageDeviceFactory

_DEFAULT_PORT: Final = "/dev/ttyACM0"


def _resolve_connection_options(
    *,
    port: str | None,
    ip_address: str | None,
) -> dict[str, str]:
    if port is not None and ip_address is not None:
        raise TypeError("Only one of `port` or `ip_address` should be provided.")
    if ip_address is not None:
        return {"ip_address": ip_address}
    return {"port": port or _DEFAULT_PORT}


def create_dc_voltage_controller(
    config: DCVoltageControllerConfig | None = None,
) -> DCVoltageController:
    """Create a DC voltage controller from normalized configuration."""
    if config is None:
        config = DCVoltageControllerConfig()
    driver = config.driver.strip().lower()
    if driver != "ons61797":
        raise ValueError(
            f"Unsupported DC voltage controller driver: {config.driver!r}."
        )
    return DCVoltageController(
        port=config.port,
        ip_address=config.ip_address,
        device_factory=config.device_factory or ONS61797Device,
    )


class DCVoltageController:
    """Provide access to a configured DC voltage device."""

    def __init__(
        self,
        *,
        port: str | None = _DEFAULT_PORT,
        ip_address: str | None = None,
        device_factory: DCVoltageDeviceFactory = ONS61797Device,
    ):
        """Initialize the controller connection settings."""
        _resolve_connection_options(port=port, ip_address=ip_address)
        self._device: DCVoltageDevice | None = None
        self._port = port
        self._ip_address = ip_address
        self._device_factory = device_factory

    def _require_device(self) -> DCVoltageDevice:
        """Return the active device or raise when disconnected."""
        if self._device is None:
            raise RuntimeError("No connection established.")
        return self._device

    def _connection_options(self) -> dict[str, str]:
        return _resolve_connection_options(
            port=self._port,
            ip_address=self._ip_address,
        )

    def _connect(self) -> None:
        connection_options = self._connection_options()
        if self._device is None:
            self._device = self._device_factory(**connection_options)
        else:
            self._device.connect(**connection_options)

    def on(self, channel: int) -> None:
        """Turn on the specified output channel."""
        with self._connection() as device:
            device.on(channel=channel)

    def off(self, channel: int) -> None:
        """Turn off the specified output channel."""
        with self._connection() as device:
            device.off(channel=channel)

    def is_output_on(self, channel: int) -> bool:
        """Return whether the specified output channel is on."""
        with self._connection() as device:
            return device.is_output_on(channel=channel)

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set the voltage for the specified channel."""
        with self._connection() as device:
            device.set_voltage(channel=channel, voltage=voltage)

    def get_voltage(self, channel: int) -> float:
        """Get the voltage for the specified channel."""
        with self._connection() as device:
            return device.get_voltage(channel=channel)

    @contextmanager
    def _connection(self) -> Iterator[DCVoltageDevice]:
        """Yield a connected device and close on exit."""
        try:
            self._connect()
            yield self._require_device()
        finally:
            if self._device is not None:
                self._device.close()

    @contextmanager
    def apply_voltages(
        self,
        requests: dict[int, tuple[float, DCVoltageProfile]],
    ) -> Iterator[DCVoltageDevice]:
        """Temporarily ramp and apply DC voltages, then safely shut down."""
        with self._connection() as device:
            try:
                for channel, (voltage, profile) in requests.items():
                    self._apply_voltage(
                        device,
                        channel=channel,
                        voltage=voltage,
                        profile=profile,
                    )
                yield device
            finally:
                for channel, (_, profile) in requests.items():
                    try:
                        if device.is_output_on(channel):
                            self._ramp_voltage(
                                device,
                                channel=channel,
                                start=device.get_voltage(channel),
                                voltage=profile.safe_voltage_v,
                                profile=profile,
                            )
                    finally:
                        device.off(channel)

    @classmethod
    def _apply_voltage(
        cls,
        device: DCVoltageDevice,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Enable one channel and ramp it to a target voltage."""
        if device.is_output_on(channel):
            start = device.get_voltage(channel)
        else:
            start = profile.safe_voltage_v
            device.set_voltage(channel, start)
            device.on(channel)
        cls._ramp_voltage(
            device,
            channel=channel,
            start=start,
            voltage=voltage,
            profile=profile,
        )

    @staticmethod
    def _ramp_voltage(
        device: DCVoltageDevice,
        *,
        channel: int,
        start: float,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Apply incremental setpoints from a start voltage to a target."""
        step = profile.ramp_rate_v_per_s * profile.update_interval_s
        direction = 1.0 if voltage >= start else -1.0
        current = float(start)
        while abs(voltage - current) > step:
            current += direction * step
            device.set_voltage(channel, current)
            time.sleep(profile.update_interval_s)
        if current != voltage:
            device.set_voltage(channel, float(voltage))
            time.sleep(profile.update_interval_s)
