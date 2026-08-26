"""Adapter for a Qblox DC voltage backend server socket protocol."""

from __future__ import annotations

import math
import socket
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from qubex.external_devices.dc_voltage.config import DCVoltageProfile
from qubex.external_devices.dc_voltage.protocol import DCVoltageDeviceFactory

_COMMAND_SET_VOLTAGE = b"\x62"
_COMMAND_GET_VOLTAGE = b"\x63"
_COMMAND_SWEEP_VOLTAGE = b"\x64"
_MIN_VOLTAGE_V = -4.0
_MAX_VOLTAGE_V = 4.0
_MAX_RAMP_RATE_V_PER_S = 1.0
_MIN_STEP_SIZE_V = 0.001 / 8.192
_MIN_WAIT_S = 0.001


def _validate_qblox_backend_ramp_settings(
    rate_v_per_s: float,
    step_size_v: float,
    wait_s: float,
) -> None:
    """Validate the Qblox server's native sweep constraints."""
    if not math.isfinite(rate_v_per_s) or not (
        0 < rate_v_per_s <= _MAX_RAMP_RATE_V_PER_S
    ):
        raise ValueError("Qblox backend ramp rate must be above 0 and at most 1 V/s.")
    if not math.isfinite(step_size_v) or step_size_v < _MIN_STEP_SIZE_V:
        raise ValueError(
            f"Qblox backend ramp step must be at least {_MIN_STEP_SIZE_V} V."
        )
    if not math.isfinite(wait_s) or wait_s < _MIN_WAIT_S:
        raise ValueError("Qblox backend ramp wait must be at least 0.001 s.")


def validate_qblox_backend_profile(profile: DCVoltageProfile) -> None:
    """Validate one resolved profile for the Qblox backend driver."""
    _validate_qblox_backend_ramp_settings(
        profile.ramp_rate_v_per_s,
        profile.ramp_step_size_v,
        profile.ramp_wait_s,
    )
    validate_qblox_backend_voltage(profile.reset_voltage_v)


def validate_qblox_backend_voltage(voltage: float) -> None:
    """Validate one target against the Qblox backend voltage range."""
    if not math.isfinite(voltage) or not _MIN_VOLTAGE_V <= voltage <= _MAX_VOLTAGE_V:
        raise ValueError("Qblox backend voltage must be between -4.0 V and 4.0 V.")


@dataclass(frozen=True)
class QbloxBackendConnectionConfig:
    """Configure a connection to a Qblox backend server."""

    host: str
    port: int
    channels: dict[int, str]
    timeout_s: float = 1200.0

    @classmethod
    def from_dict(
        cls,
        device_id: str,
        params: Mapping[str, object],
        device_channels: Sequence[int] | None = None,
    ) -> QbloxBackendConnectionConfig:
        """Parse Qblox backend endpoint and channel names."""
        known = {"host", "port", "timeout_s", "device_names"}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"Unknown Qblox server settings: {sorted(unknown)}.")
        host = params.get("host")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("Qblox server `host` must be a non-empty string.")
        port = params.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Qblox server `port` must be between 1 and 65535.")
        timeout_s = params.get("timeout_s", 1200.0)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("Qblox server `timeout_s` must be numeric.")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Qblox server `timeout_s` must be positive and finite.")
        channels = _parse_device_names(device_id, params, device_channels)
        return cls(
            host=host,
            port=port,
            channels=channels,
            timeout_s=float(timeout_s),
        )


def _parse_device_names(
    device_id: str,
    params: Mapping[str, object],
    device_channels: Sequence[int] | None,
) -> dict[int, str]:
    """Derive backend device names from the device id or an explicit mapping."""
    raw_names = params.get("device_names")
    if raw_names is None:
        if not device_id:
            raise ValueError("Qblox server requires a non-empty device name.")
        if "\x00" in device_id:
            raise ValueError("Qblox server device names must not contain NUL bytes.")
        if not device_channels:
            raise ValueError(
                "Qblox server requires a non-empty device `channels` list."
            )
        return {channel: f"{device_id}-{channel}" for channel in device_channels}
    if not isinstance(raw_names, Mapping) or not raw_names:
        raise ValueError("Qblox server `device_names` must be a non-empty mapping.")
    names: dict[int, str] = {}
    for channel, device_name in raw_names.items():
        if type(channel) is not int:
            raise ValueError("Qblox server `device_names` keys must be integers.")
        if not isinstance(device_name, str) or not device_name:
            raise ValueError("Qblox server device names must be non-empty strings.")
        if "\x00" in device_name:
            raise ValueError("Qblox server device names must not contain NUL bytes.")
        names[channel] = device_name
    if len(set(names.values())) != len(names):
        raise ValueError("Qblox server device names must be unique across channels.")
    if device_channels is not None:
        missing = sorted(set(device_channels) - set(names))
        if missing:
            raise ValueError(
                f"Qblox server `device_names` is missing device channels {missing}."
            )
    return names


def create_qblox_backend_device_factory(
    device_id: str,
    params: Mapping[str, object],
    device_channels: Sequence[int] | None = None,
) -> DCVoltageDeviceFactory:
    """Create a Qblox backend server device factory."""
    config = QbloxBackendConnectionConfig.from_dict(device_id, params, device_channels)
    return partial(
        QbloxBackendClient,
        host=config.host,
        port=config.port,
        channels=config.channels,
        timeout_s=config.timeout_s,
    )


class QbloxBackendClient:
    """Use a Qblox backend server as the single USB device owner."""

    def __init__(
        self,
        host: str,
        port: int,
        channels: Mapping[int, str],
        timeout_s: float = 1200.0,
        *,
        socket_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Open one client connection to the configured backend server."""
        self._channels = dict(channels)
        factory = socket.create_connection if socket_factory is None else socket_factory
        self._socket = factory((host, port), timeout=timeout_s)

    @property
    def supports_native_ramp(self) -> bool:
        """Return that the backend can execute one complete sweep request."""
        return True

    @property
    def supports_output_switch(self) -> bool:
        """Return that the D5a backend has no physical output switch."""
        return False

    def close(self) -> None:
        """Close the backend socket without changing output voltages."""
        self._socket.close()

    def on(self, channel: int) -> None:
        """Reject physical output switching, which the backend lacks."""
        self._device_name(channel)
        raise NotImplementedError("Qblox backend has no physical output switch.")

    def off(self, channel: int) -> None:
        """Reject physical output switching, which the backend lacks."""
        self._device_name(channel)
        raise NotImplementedError("Qblox backend has no physical output switch.")

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set one backend device voltage immediately in volts."""
        self._validate_voltage(voltage)
        request = (
            _COMMAND_SET_VOLTAGE
            + self._device_name(channel)
            + b"\x00"
            + struct.pack("<d", voltage)
        )
        self._socket.sendall(request)
        self._require_success("set voltage")

    def get_voltage(self, channel: int) -> float:
        """Return one backend device's configured voltage in volts."""
        request = _COMMAND_GET_VOLTAGE + self._device_name(channel) + b"\x00"
        self._socket.sendall(request)
        self._require_success("get voltage")
        return struct.unpack("<d", self._recv_exact(8))[0]

    def is_output_on(self, channel: int) -> bool:
        """Return true because the backend cannot switch D5a outputs off."""
        self._device_name(channel)
        return True

    def ramp_voltage(
        self,
        channel: int,
        start_voltage: float,
        target_voltage: float,
        rate_v_per_s: float,
        step_size_v: float,
        wait_s: float,
    ) -> None:
        """Request one complete backend-side voltage sweep."""
        self._validate_voltage(start_voltage)
        self._validate_voltage(target_voltage)
        _validate_qblox_backend_ramp_settings(rate_v_per_s, step_size_v, wait_s)
        request = (
            _COMMAND_SWEEP_VOLTAGE
            + self._device_name(channel)
            + b"\x00"
            + struct.pack(
                "<ddddd",
                start_voltage,
                target_voltage,
                rate_v_per_s,
                step_size_v,
                wait_s,
            )
        )
        self._socket.sendall(request)
        self._require_success("ramp voltage")

    def _device_name(self, channel: int) -> bytes:
        """Resolve a Qubex channel to its backend device identifier."""
        try:
            return self._channels[channel].encode()
        except KeyError:
            raise ValueError(
                f"Qblox server channel {channel} is not configured."
            ) from None

    @staticmethod
    def _validate_voltage(voltage: float) -> None:
        """Validate the backend's standard bipolar voltage range."""
        validate_qblox_backend_voltage(voltage)

    def _require_success(self, operation: str, *, status: bytes | None = None) -> None:
        """Require the backend's one-byte zero success response."""
        resolved_status = self._recv_exact(1) if status is None else status
        if resolved_status != b"\x00":
            raise RuntimeError(f"Qblox backend rejected {operation}.")

    def _recv_exact(self, size: int) -> bytes:
        """Receive exactly one fixed-size backend response."""
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise ConnectionError(
                    "Qblox backend closed before completing a response."
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
