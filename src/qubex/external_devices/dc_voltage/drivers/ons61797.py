"""Qubex external-device adapter for the vendored ONS61797 client."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from qubex.external_devices.dc_voltage.protocol import DCVoltageDeviceFactory
from qubex.third_party.ons61797 import ONS61797

_MIN_VOLTAGE_V = 0.0
_MAX_VOLTAGE_V = 4.0


@dataclass(frozen=True)
class ONS61797ConnectionConfig:
    """Configure an ONS61797 serial or network connection."""

    port: str | None = "/dev/ttyACM0"
    ip_address: str | None = None

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> ONS61797ConnectionConfig:
        """Parse driver-specific connection settings."""
        unknown = set(params) - {"port", "ip_address"}
        if unknown:
            raise ValueError(f"Unknown ONS61797 settings: {sorted(unknown)}.")
        port = params.get("port")
        ip_address = params.get("ip_address")
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
    device_id: str,
    params: Mapping[str, object],
    device_channels: Sequence[int] | None = None,
) -> DCVoltageDeviceFactory:
    """Create an ONS61797 device factory from opaque driver settings."""
    del device_id, device_channels  # ONS61797 addresses outputs by channel.
    config = ONS61797ConnectionConfig.from_dict(params)
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
        output_mode = self._client.get_output_mode()
        if output_mode != 0:
            self._client.close()
            raise RuntimeError(
                "ONS61797 must use independent output mode (OMD 0) for "
                "per-channel control."
            )

    @property
    def supports_native_ramp(self) -> bool:
        """Return that Qubex must generate ONS61797 ramp setpoints."""
        return False

    @property
    def supports_output_switch(self) -> bool:
        """Return that ONS61797 supports physical output switching."""
        return True

    def close(self) -> None:
        """Close the underlying ONS61797 client."""
        self._client.close()

    def on(self, channel: int) -> None:
        """Turn on one output channel after checking its stored setpoint."""
        self._validate_channel(channel)
        stored = self._client.get_voltage(channel=channel)
        if not math.isfinite(stored) or not _MIN_VOLTAGE_V <= stored <= _MAX_VOLTAGE_V:
            raise ValueError(
                f"ONS61797 channel {channel} stored setpoint is {stored} V, "
                f"outside {_MIN_VOLTAGE_V} V to {_MAX_VOLTAGE_V} V. Set a "
                "valid voltage before turning the output on."
            )
        self._client.on(channel=channel)

    def off(self, channel: int) -> None:
        """Turn off one output channel."""
        self._validate_channel(channel)
        self._client.off(channel=channel)

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for one output channel within the allowed range."""
        self._validate_channel(channel)
        if (
            not math.isfinite(voltage)
            or not _MIN_VOLTAGE_V <= voltage <= _MAX_VOLTAGE_V
        ):
            raise ValueError(
                f"ONS61797 voltage must be between {_MIN_VOLTAGE_V} V and "
                f"{_MAX_VOLTAGE_V} V, got {voltage} V."
            )
        self._client.set_voltage(channel=channel, voltage=voltage)

    def get_voltage(self, channel: int) -> float:
        """Return voltage for one output channel."""
        self._validate_channel(channel)
        return self._client.get_voltage(channel=channel)

    def is_output_on(self, channel: int) -> bool:
        """Return whether one output channel is on."""
        self._validate_channel(channel)
        return self._client.get_output_state(channel=channel) == 1

    @staticmethod
    def _validate_channel(channel: int) -> None:
        """Require one documented ONS61797 channel number."""
        if type(channel) is not int or not 1 <= channel <= 16:
            raise ValueError("ONS61797 channel must be between 1 and 16.")

    def ramp_voltage(
        self,
        channel: int,
        start_voltage: float,
        target_voltage: float,
        rate_v_per_s: float,
        step_size_v: float,
        wait_s: float,
    ) -> None:
        """Reject native ramping, which this adapter does not expose."""
        raise NotImplementedError("ONS61797 does not expose native ramping.")
