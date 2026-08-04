"""External DC voltage controller."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

from .config import DCVoltageControllerConfig, DCVoltageProfile
from .protocol import DCVoltageDevice, DCVoltageDeviceFactory
from .registry import DC_VOLTAGE_DRIVER_REGISTRY


class DCVoltageExitMode(str, Enum):
    """Define controller behavior when a voltage context exits."""

    IDLE = "idle"
    HOLD = "hold"


def create_dc_voltage_controller(
    config: DCVoltageControllerConfig | None = None,
) -> DCVoltageController:
    """Create a DC voltage controller from normalized configuration."""
    if config is None:
        config = DCVoltageControllerConfig()
    if config.device_factory is not None:
        return DCVoltageController(device_factory=config.device_factory)
    driver_name = config.driver.strip().lower()
    try:
        driver_factory = DC_VOLTAGE_DRIVER_REGISTRY[driver_name]
    except KeyError:
        raise ValueError(
            f"Unsupported DC voltage controller driver: {config.driver!r}."
        ) from None
    return DCVoltageController(
        device_factory=driver_factory(
            config.device_id or "",
            config.params,
            config.channels,
        )
    )


class DCVoltageController:
    """Provide access to a configured DC voltage device."""

    def __init__(
        self,
        *,
        device_factory: DCVoltageDeviceFactory,
    ):
        """Initialize the controller with a connected-device factory."""
        self._device_factory = device_factory

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

    def apply_voltage(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Enable one channel and ramp it to a target voltage."""
        with self._connection() as device:
            self._apply_voltage(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )

    def apply_voltage_immediately(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Enable one channel and apply a target without ramping."""
        with self._connection() as device:
            is_output_on = device.is_output_on(channel)
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )
            if not is_output_on:
                device.on(channel)

    def idle(
        self,
        *,
        channel: int,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one channel to its idle voltage."""
        with self._connection() as device:
            self._ramp_to_idle(
                device,
                channel=channel,
                profile=profile,
            )

    @contextmanager
    def _connection(self) -> Iterator[DCVoltageDevice]:
        """Yield a connected device and close on exit."""
        device = self._device_factory()
        try:
            yield device
        finally:
            device.close()

    @contextmanager
    def apply_voltages(
        self,
        requests: dict[int, tuple[float, DCVoltageProfile]],
        *,
        exit_modes: dict[int, DCVoltageExitMode] | None = None,
    ) -> Iterator[DCVoltageDevice]:
        """Apply DC voltages and return each channel to idle unless told to hold."""
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
                    mode = (
                        exit_modes.get(channel, DCVoltageExitMode.IDLE)
                        if exit_modes is not None
                        else DCVoltageExitMode.IDLE
                    )
                    if mode is DCVoltageExitMode.IDLE:
                        self._ramp_to_idle(device, channel=channel, profile=profile)

    def _apply_voltage(
        self,
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
            start = profile.idle_voltage_v
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=start,
                profile=profile,
            )
            device.on(channel)
        self._ramp_voltage(
            device,
            channel=channel,
            start=start,
            voltage=voltage,
            profile=profile,
        )

    def _ramp_to_idle(
        self,
        device: DCVoltageDevice,
        *,
        channel: int,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one enabled output back to its idle voltage."""
        if device.is_output_on(channel):
            self._ramp_voltage(
                device,
                channel=channel,
                start=device.get_voltage(channel),
                voltage=profile.idle_voltage_v,
                profile=profile,
            )

    def _ramp_voltage(
        self,
        device: DCVoltageDevice,
        *,
        channel: int,
        start: float,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Apply incremental setpoints from a start voltage to a target."""
        step = profile.ramp_rate_v_per_s * profile.update_interval_s
        if start == voltage:
            return
        if device.supports_native_ramp:
            current = start
            for _ in range(profile.max_set_attempts):
                device.ramp_voltage(
                    channel,
                    current,
                    voltage,
                    profile.ramp_rate_v_per_s,
                    step,
                    profile.update_interval_s,
                )
                current = device.get_voltage(channel)
                if abs(current - voltage) <= profile.readback_tolerance_v:
                    return
            raise RuntimeError(
                f"DC voltage channel {channel} failed to reach {voltage} V "
                f"within tolerance {profile.readback_tolerance_v} V after "
                f"{profile.max_set_attempts} native ramp attempts."
            )
        direction = 1.0 if voltage >= start else -1.0
        current = float(start)
        while abs(voltage - current) > step:
            current += direction * step
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=current,
                profile=profile,
            )
            time.sleep(profile.update_interval_s)
        if current != voltage:
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=float(voltage),
                profile=profile,
            )
            time.sleep(profile.update_interval_s)

    def _set_voltage_verified(
        self,
        device: DCVoltageDevice,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Set one voltage and require readback within configured tolerance."""
        for _ in range(profile.max_set_attempts):
            device.set_voltage(channel, voltage)
            if (
                abs(device.get_voltage(channel) - voltage)
                <= profile.readback_tolerance_v
            ):
                return
        raise RuntimeError(
            f"DC voltage channel {channel} failed to reach {voltage} V "
            f"within tolerance {profile.readback_tolerance_v} V after "
            f"{profile.max_set_attempts} attempts."
        )
