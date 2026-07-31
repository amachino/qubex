"""External DC voltage device contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class DCVoltageDevice(Protocol):
    """Define the operations required from a DC voltage source."""

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

    def is_output_on(self, channel: int) -> bool:
        """Return whether one output channel is on."""
        ...


DCVoltageDeviceFactory = Callable[..., DCVoltageDevice]
