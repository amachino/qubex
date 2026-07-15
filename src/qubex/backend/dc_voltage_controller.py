"""DC voltage control helpers for backend-managed DC devices."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol

from qubex.third_party.ons61797 import ONS61797

DEFAULT_PORT: Final = "/dev/ttyACM0"
PORT: Final = DEFAULT_PORT


class DCVoltageDevice(Protocol):
    """Protocol for DC voltage source devices."""

    def connect(
        self,
        port: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Connect to the device."""
        ...

    def close(self) -> None:
        """Close the device connection."""
        ...

    def on(self, channel: int) -> None:
        """Turn on one output channel."""
        ...

    def off(self, channel: int) -> None:
        """Turn off one output channel."""
        ...

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for one output channel."""
        ...

    def get_voltage(self, channel: int) -> float:
        """Return voltage for one output channel."""
        ...

    def get_output_state(self, channel: int) -> int:
        """Return output state for one output channel."""
        ...

    def get_device_information(self) -> str:
        """Return device information."""
        ...

    def reset(self) -> None:
        """Reset the device."""
        ...


DCVoltageDeviceFactory = Callable[..., DCVoltageDevice]
DCVoltageControllerDriver = str


@dataclass(frozen=True)
class DCVoltageControllerConfig:
    """Configuration for a DC voltage controller."""

    driver: DCVoltageControllerDriver = "ons61797"
    port: str | None = None
    ip_address: str | None = None
    device_factory: DCVoltageDeviceFactory | None = None


def _resolve_connection_options(
    *,
    port: str | None,
    ip_address: str | None,
) -> dict[str, str]:
    if port is not None and ip_address is not None:
        raise TypeError("Only one of `port` or `ip_address` should be provided.")
    if ip_address is not None:
        return {"ip_address": ip_address}
    return {"port": port or DEFAULT_PORT}


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
        device_factory=config.device_factory or ONS61797,
    )


@contextmanager
def dc_voltage(
    voltages: dict[int, float],
    *,
    port: str | None = None,
    ip_address: str | None = None,
    device_factory: DCVoltageDeviceFactory = ONS61797,
) -> Iterator[DCVoltageDevice]:
    """Temporarily apply DC voltages and restore originals on exit."""
    device: DCVoltageDevice | None = None
    original_voltages: dict[int, float] = {}
    try:
        device = device_factory(
            **_resolve_connection_options(
                port=port,
                ip_address=ip_address,
            )
        )
        for channel, voltage in voltages.items():
            original_voltages[channel] = device.get_voltage(channel)
            device.set_voltage(channel, voltage)
            device.on(channel)
        yield device
    finally:
        if device is not None:
            for channel, voltage in original_voltages.items():
                device.set_voltage(channel, voltage)
                device.off(channel)
            device.close()


def with_connection(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap calls with a temporary DC device connection."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            self._connect()
            return func(self, *args, **kwargs)
        finally:
            if self._device is not None:
                self._device.close()

    return wrapper


class DCVoltageController:
    """Singleton controller for DC voltage device access."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls) -> DCVoltageController:
        """Return the shared controller instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        *,
        port: str | None = DEFAULT_PORT,
        ip_address: str | None = None,
        device_factory: DCVoltageDeviceFactory = ONS61797,
    ):
        """Initialize the controller if not already initialized."""
        if self._initialized:
            if (
                port != DEFAULT_PORT
                or ip_address is not None
                or device_factory is not ONS61797
            ):
                self.configure(
                    port=port,
                    ip_address=ip_address,
                    device_factory=device_factory,
                )
            return
        self._device: DCVoltageDevice | None = None
        self._port = port
        self._ip_address = ip_address
        self._device_factory = device_factory
        self._initialized = True

    def __del__(self):
        """Close the device connection on deletion."""
        if self._device is not None:
            self._device.close()

    @property
    def ons61797(self) -> DCVoltageDevice:
        """Return the active device connection."""
        if self._device is None:
            raise RuntimeError("No connection established.")
        return self._device

    @property
    def device(self) -> DCVoltageDevice:
        """Return the active device connection."""
        return self.ons61797

    @classmethod
    def reset_shared(cls) -> None:
        """Reset the shared controller instance."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False

    def configure(
        self,
        *,
        port: str | None = DEFAULT_PORT,
        ip_address: str | None = None,
        device_factory: DCVoltageDeviceFactory = ONS61797,
    ) -> None:
        """Configure connection options for subsequent device calls."""
        _resolve_connection_options(port=port, ip_address=ip_address)
        self._port = port
        self._ip_address = ip_address
        self._device_factory = device_factory

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

    def close(self) -> None:
        """Close the active device connection."""
        if self._device is not None:
            self._device.close()

    @with_connection
    def on(self, channel: int) -> None:
        """Turn on the specified output channel."""
        self.ons61797.on(channel=channel)

    @with_connection
    def off(self, channel: int) -> None:
        """Turn off the specified output channel."""
        self.ons61797.off(channel=channel)

    @with_connection
    def get_output_state(self, channel: int) -> int:
        """Get the output state of the specified channel."""
        return self.ons61797.get_output_state(channel=channel)

    @with_connection
    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set the voltage for the specified channel."""
        self.ons61797.set_voltage(channel=channel, voltage=voltage)

    @with_connection
    def get_voltage(self, channel: int) -> float:
        """Get the voltage for the specified channel."""
        return self.ons61797.get_voltage(channel=channel)

    @with_connection
    def get_device_information(self) -> str:
        """Return device information from the controller."""
        return self.ons61797.get_device_information()

    @with_connection
    def reset(self) -> None:
        """Reset the device settings."""
        self.ons61797.reset()

    @contextmanager
    def connection(self) -> Iterator[DCVoltageDevice]:
        """Yield a connected device and close on exit."""
        try:
            self._connect()
            if self._device is None:
                raise RuntimeError("No connection established.")
            yield self._device
        finally:
            if self._device is not None:
                self._device.close()

    @contextmanager
    def apply_voltages(self, voltages: dict[int, float]) -> Iterator[DCVoltageDevice]:
        """Temporarily apply DC voltages and restore originals on exit."""
        original_voltages: dict[int, float] = {}
        with self.connection() as device:
            try:
                for channel, voltage in voltages.items():
                    original_voltages[channel] = device.get_voltage(channel)
                    device.set_voltage(channel, voltage)
                    device.on(channel)
                yield device
            finally:
                for channel, voltage in original_voltages.items():
                    device.set_voltage(channel, voltage)
                    device.off(channel)
