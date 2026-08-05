"""External DC voltage controller."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from .config import DCVoltageControllerConfig, DCVoltageProfile
from .protocol import DCVoltageDevice, DCVoltageDeviceFactory
from .registry import DC_VOLTAGE_DRIVER_REGISTRY

logger = logging.getLogger(__name__)


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
        """Ramp one enabled channel to a target voltage."""
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
        """Apply a target to one enabled channel without ramping."""
        with self._connection() as device:
            if not device.is_output_on(channel):
                raise RuntimeError(
                    f"DC voltage channel {channel} output is off. Initialize it "
                    "with `reset_dc_voltages()` before applying a voltage."
                )
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )

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

    def apply_channels(
        self,
        requests: Mapping[int, tuple[float, DCVoltageProfile]],
    ) -> None:
        """Ramp many channels to their target voltages on one connection."""
        with self._connection() as device:
            for channel, (voltage, profile) in requests.items():
                self._apply_voltage(
                    device,
                    channel=channel,
                    voltage=voltage,
                    profile=profile,
                )

    def reset_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Bring channels to their reset voltages with outputs on."""
        with self._connection() as device:
            for channel, profile in profiles.items():
                if device.is_output_on(channel):
                    self._ramp_voltage(
                        device,
                        channel=channel,
                        start=device.get_voltage(channel),
                        voltage=profile.reset_voltage_v,
                        profile=profile,
                    )
                    continue
                self._set_voltage_verified(
                    device,
                    channel=channel,
                    voltage=profile.reset_voltage_v,
                    profile=profile,
                )
                device.on(channel)
                logger.info(
                    "DC channel %d: output switched on at %+.3f V.",
                    channel,
                    profile.reset_voltage_v,
                )

    def turn_off_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Ramp outputs to reset voltage and switch them off when supported."""
        with self._connection() as device:
            for channel, profile in profiles.items():
                if not device.is_output_on(channel):
                    logger.info("DC channel %d: output is already off.", channel)
                    continue
                self._ramp_voltage(
                    device,
                    channel=channel,
                    start=device.get_voltage(channel),
                    voltage=profile.reset_voltage_v,
                    profile=profile,
                )
                if device.supports_output_switch:
                    device.off(channel)
                    logger.info("DC channel %d: output switched off.", channel)
                else:
                    logger.info(
                        "DC channel %d: no physical output switch; held at %+.3f V.",
                        channel,
                        profile.reset_voltage_v,
                    )

    def read_channels(
        self,
        channels: Sequence[int],
    ) -> dict[int, tuple[float, bool]]:
        """Read voltage and output state for many channels on one connection."""
        with self._connection() as device:
            return {
                channel: (
                    device.get_voltage(channel),
                    device.is_output_on(channel),
                )
                for channel in channels
            }

    def idle_channels(
        self,
        profiles: Mapping[int, DCVoltageProfile],
    ) -> None:
        """Ramp many channels back to their idle voltages on one connection."""
        with self._connection() as device:
            for channel, profile in profiles.items():
                self._ramp_to_idle(device, channel=channel, profile=profile)

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
    ) -> Iterator[None]:
        """
        Apply DC voltages and return each channel to idle on exit.

        The device connection is open only while voltages are being changed;
        no connection is held while the caller's block runs.
        """
        try:
            self.apply_channels(requests)
            yield
        finally:
            if requests:
                self.idle_channels(
                    {channel: profile for channel, (_, profile) in requests.items()}
                )

    def _apply_voltage(
        self,
        device: DCVoltageDevice,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one enabled channel to a target voltage."""
        if not device.is_output_on(channel):
            raise RuntimeError(
                f"DC voltage channel {channel} output is off. Initialize it "
                "with `reset_dc_voltages()` before applying a voltage."
            )
        start = device.get_voltage(channel)
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
        else:
            logger.info(
                "DC channel %d: output is off; not ramped, its stored "
                "setpoint stays at %+.3f V.",
                channel,
                device.get_voltage(channel),
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
        step = profile.ramp_step_size_v
        wait = max(step / profile.ramp_rate_v_per_s, profile.ramp_wait_s)
        if start == voltage:
            return
        logger.info(
            "DC channel %d: ramping %+.3f V -> %+.3f V at %.3f V/s.",
            channel,
            start,
            voltage,
            profile.ramp_rate_v_per_s,
        )
        if device.supports_native_ramp:
            current = start
            for _ in range(profile.max_set_attempts):
                device.ramp_voltage(
                    channel,
                    current,
                    voltage,
                    profile.ramp_rate_v_per_s,
                    step,
                    profile.ramp_wait_s,
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
            time.sleep(wait)
        if current != voltage:
            self._set_voltage_verified(
                device,
                channel=channel,
                voltage=float(voltage),
                profile=profile,
            )
            time.sleep(wait)

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
