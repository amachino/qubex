"""External DC voltage device contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class DCVoltageDevice(Protocol):
    """Define the operations required from a DC voltage source."""

    @property
    def supports_native_ramp(self) -> bool:
        """Return whether the device can execute a complete ramp atomically."""
        ...

    @property
    def supports_output_switch(self) -> bool:
        """Return whether the device can physically switch an output off."""
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

    def ramp_voltage(
        self,
        channel: int,
        start_voltage: float,
        target_voltage: float,
        rate_v_per_s: float,
        step_size_v: float,
        wait_s: float,
    ) -> None:
        """Ramp one channel using device-native execution."""
        ...


DCVoltageDeviceFactory = Callable[[], DCVoltageDevice]
