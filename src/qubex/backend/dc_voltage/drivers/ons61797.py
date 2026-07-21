"""Qubex adapter for the vendored ONS61797 client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qubex.third_party.ons61797 import ONS61797


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

    def connect(
        self,
        port: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Connect the underlying ONS61797 client."""
        self._client.connect(port=port, ip_address=ip_address)

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
