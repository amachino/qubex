"""Qubex external-device adapter for the vendored ONS61797 client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from qubex.external_devices.dc_voltage.protocol import DCVoltageDeviceFactory
from qubex.third_party.ons61797 import ONS61797


@dataclass(frozen=True)
class ONS61797ConnectionConfig:
    """Configure an ONS61797 serial or network connection."""

    port: str | None = "/dev/ttyACM0"
    ip_address: str | None = None

    @classmethod
    def from_dict(cls, connection: Mapping[str, object]) -> ONS61797ConnectionConfig:
        """Parse driver-specific connection settings."""
        unknown = set(connection) - {"port", "ip_address"}
        if unknown:
            raise ValueError(
                f"Unknown ONS61797 connection settings: {sorted(unknown)}."
            )
        port = connection.get("port")
        ip_address = connection.get("ip_address")
        if port is not None and not isinstance(port, str):
            raise TypeError("ONS61797 `port` must be a string.")
        if ip_address is not None and not isinstance(ip_address, str):
            raise TypeError("ONS61797 `ip_address` must be a string.")
        if port is not None and ip_address is not None:
            raise TypeError("Only one of `port` or `ip_address` may be provided.")
        if ip_address is not None:
            return cls(port=None, ip_address=ip_address)
        return cls(port=port or "/dev/ttyACM0")


def create_ons61797_device_factory(
    connection: Mapping[str, object],
) -> DCVoltageDeviceFactory:
    """Create an ONS61797 device factory from opaque connection settings."""
    config = ONS61797ConnectionConfig.from_dict(connection)
    return partial(
        ONS61797Device,
        port=config.port,
        ip_address=config.ip_address,
    )


class ONS61797Device:
    """Adapt the vendored ONS61797 client to the DC device protocol."""

    def __init__(
        self,
        port: str | None = None,
        ip_address: str | None = None,
        *,
        client_factory: Callable[..., Any] = ONS61797,
    ) -> None:
        """Create an ONS61797 client using serial or network transport."""
        self._client = client_factory(port=port, ip_address=ip_address)

    def close(self) -> None:
        """Close the underlying ONS61797 client."""
        self._client.close()

    def on(self, channel: int) -> None:
        """Turn on one output channel."""
        self._client.on(channel=channel)

    def off(self, channel: int) -> None:
        """Turn off one output channel."""
        self._client.off(channel=channel)

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for one output channel."""
        self._client.set_voltage(channel=channel, voltage=voltage)

    def get_voltage(self, channel: int) -> float:
        """Return voltage for one output channel."""
        return self._client.get_voltage(channel=channel)

    def is_output_on(self, channel: int) -> bool:
        """Return whether one output channel is on."""
        return self._client.get_output_state(channel=channel) == 1
