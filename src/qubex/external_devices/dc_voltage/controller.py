"""External DC voltage controller."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
        driver_spec = DC_VOLTAGE_DRIVER_REGISTRY[driver_name]
    except KeyError:
        raise ValueError(
            f"Unsupported DC voltage controller driver: {config.driver!r}."
        ) from None
    if driver_spec.validate_profile is not None:
        for mux_index in config.muxes:
            driver_spec.validate_profile(config.resolve_voltage_profile(mux_index))
    return DCVoltageController(
        device_factory=driver_spec.create_device_factory(
            config.device_id or "",
            config.params,
            config.channels,
        ),
        validate_voltage=driver_spec.validate_voltage,
    )


class DCVoltageConnection:
    """Operate a DC voltage device through one open connection."""

    def __init__(
        self,
        *,
        device: DCVoltageDevice,
        apply_voltage: Callable[[int, float, DCVoltageProfile], None],
        idle: Callable[[int, DCVoltageProfile], None],
        apply_channels: Callable[[Mapping[int, tuple[float, DCVoltageProfile]]], None],
        reset_channels: Callable[[Mapping[int, DCVoltageProfile]], None],
        turn_off_channels: Callable[[Mapping[int, DCVoltageProfile]], None],
        idle_channels: Callable[[Mapping[int, DCVoltageProfile]], None],
    ) -> None:
        """Bind controller operations to one connected device."""
        self._device = device
        self._apply_voltage = apply_voltage
        self._idle = idle
        self._apply_channels = apply_channels
        self._reset_channels = reset_channels
        self._turn_off_channels = turn_off_channels
        self._idle_channels = idle_channels

    def apply_voltage_and_read(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> tuple[float, bool]:
        """Ramp one channel and return readback without reconnecting."""
        self._apply_voltage(channel, voltage, profile)
        return self.read_channel(channel)

    def read_channel(self, channel: int) -> tuple[float, bool]:
        """Return one channel's voltage and output state."""
        return self._device.get_voltage(channel), self._device.is_output_on(channel)

    def read_channels(
        self,
        channels: Sequence[int],
    ) -> dict[int, tuple[float, bool]]:
        """Return many channel states without reconnecting."""
        return {
            channel: (
                self._device.get_voltage(channel),
                self._device.is_output_on(channel),
            )
            for channel in channels
        }

    def idle(self, *, channel: int, profile: DCVoltageProfile) -> None:
        """Ramp one channel to idle without reconnecting."""
        self._idle(channel, profile)

    def apply_channels(
        self,
        requests: Mapping[int, tuple[float, DCVoltageProfile]],
    ) -> None:
        """Ramp many channels without reconnecting."""
        self._apply_channels(requests)

    def reset_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Reset many channels without reconnecting."""
        self._reset_channels(profiles)

    def turn_off_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Shut down many channels without reconnecting."""
        self._turn_off_channels(profiles)

    def idle_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Idle many channels without reconnecting."""
        self._idle_channels(profiles)


class DCVoltageController:
    """Provide access to a configured DC voltage device."""

    def __init__(
        self,
        *,
        device_factory: DCVoltageDeviceFactory,
        validate_voltage: Callable[[float], None] | None = None,
    ):
        """Initialize the controller with a connected-device factory."""
        self._device_factory = device_factory
        self._validate_voltage = validate_voltage

    @contextmanager
    def connected(self) -> Iterator[DCVoltageConnection]:
        """Yield operations bound to one open device connection."""
        with self._connection() as device:
            yield DCVoltageConnection(
                device=device,
                apply_voltage=lambda channel, voltage, profile: self._apply_voltage(
                    device,
                    channel=channel,
                    voltage=voltage,
                    profile=profile,
                ),
                idle=lambda channel, profile: self._ramp_to_idle(
                    device,
                    channel=channel,
                    profile=profile,
                ),
                apply_channels=lambda requests: self._apply_channels(
                    device,
                    requests,
                ),
                reset_channels=lambda profiles: self._reset_channels(
                    device,
                    profiles,
                ),
                turn_off_channels=lambda profiles: self._turn_off_channels(
                    device,
                    profiles,
                ),
                idle_channels=lambda profiles: self._idle_channels(
                    device,
                    profiles,
                ),
            )

    def validate_voltage(self, voltage: float) -> None:
        """Validate one target voltage without opening a device connection."""
        self._validate_voltages((voltage,))

    def apply_voltage(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one enabled channel to a target voltage."""
        self._validate_voltages((voltage,))
        with self._connection() as device:
            self._apply_voltage(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )

    def apply_voltage_and_read(
        self,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> tuple[float, bool]:
        """Ramp one channel and return its readback on the same connection."""
        self._validate_voltages((voltage,))
        with self._connection() as device:
            self._apply_voltage(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )
            return device.get_voltage(channel), device.is_output_on(channel)

    def idle(
        self,
        *,
        channel: int,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one channel to its idle voltage."""
        self._validate_voltages((profile.idle_voltage_v,))
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
        self._validate_voltages(voltage for voltage, _ in requests.values())
        with self._connection() as device:
            self._apply_channels(device, requests)

    def reset_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Bring channels to their reset voltages with outputs on."""
        self._validate_voltages(
            profile.reset_voltage_v for profile in profiles.values()
        )
        with self._connection() as device:
            self._reset_channels(device, profiles)

    def turn_off_channels(self, profiles: Mapping[int, DCVoltageProfile]) -> None:
        """Ramp outputs to reset voltage and switch them off when supported."""
        self._validate_voltages(
            profile.reset_voltage_v for profile in profiles.values()
        )
        with self._connection() as device:
            self._turn_off_channels(device, profiles)

    def read_channels(
        self,
        channels: Sequence[int],
    ) -> dict[int, tuple[float, bool]]:
        """Read voltage and output state for many channels on one connection."""
        if not channels:
            return {}
        with self._connection() as device:
            return self._read_channels(device, channels)

    def idle_channels(
        self,
        profiles: Mapping[int, DCVoltageProfile],
    ) -> None:
        """Ramp many channels back to their idle voltages on one connection."""
        self._validate_voltages(profile.idle_voltage_v for profile in profiles.values())
        with self._connection() as device:
            self._idle_channels(device, profiles)

    def _apply_channels(
        self,
        device: DCVoltageDevice,
        requests: Mapping[int, tuple[float, DCVoltageProfile]],
    ) -> None:
        """Ramp many channels through one connected device."""
        self._validate_voltages(voltage for voltage, _ in requests.values())
        for channel, (voltage, profile) in requests.items():
            self._apply_voltage(
                device,
                channel=channel,
                voltage=voltage,
                profile=profile,
            )

    def _reset_channels(
        self,
        device: DCVoltageDevice,
        profiles: Mapping[int, DCVoltageProfile],
    ) -> None:
        """Reset many channels through one connected device."""
        self._validate_voltages(
            profile.reset_voltage_v for profile in profiles.values()
        )
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

    def _turn_off_channels(
        self,
        device: DCVoltageDevice,
        profiles: Mapping[int, DCVoltageProfile],
    ) -> None:
        """Shut down many channels through one connected device."""
        self._validate_voltages(
            profile.reset_voltage_v for profile in profiles.values()
        )
        for channel, profile in profiles.items():
            if not device.is_output_on(channel):
                self._set_voltage_verified(
                    device,
                    channel=channel,
                    voltage=profile.reset_voltage_v,
                    profile=profile,
                )
                logger.info(
                    "DC channel %d: output remains off with its setpoint "
                    "reset to %+.3f V.",
                    channel,
                    profile.reset_voltage_v,
                )
                continue
            start = device.get_voltage(channel)
            try:
                self._validate_voltages((start,))
            except ValueError:
                if not device.supports_output_switch:
                    raise
                device.off(channel)
                logger.warning(
                    "DC channel %d: output switched off before recovering "
                    "unsafe readback %+.3f V.",
                    channel,
                    start,
                )
                self._set_voltage_verified(
                    device,
                    channel=channel,
                    voltage=profile.reset_voltage_v,
                    profile=profile,
                )
                continue
            self._ramp_voltage(
                device,
                channel=channel,
                start=start,
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

    @staticmethod
    def _read_channels(
        device: DCVoltageDevice,
        channels: Sequence[int],
    ) -> dict[int, tuple[float, bool]]:
        """Read many channels through one connected device."""
        return {
            channel: (
                device.get_voltage(channel),
                device.is_output_on(channel),
            )
            for channel in channels
        }

    def _idle_channels(
        self,
        device: DCVoltageDevice,
        profiles: Mapping[int, DCVoltageProfile],
    ) -> None:
        """Idle many channels through one connected device."""
        self._validate_voltages(profile.idle_voltage_v for profile in profiles.values())
        for channel, profile in profiles.items():
            self._ramp_to_idle(device, channel=channel, profile=profile)

    def _validate_voltages(self, voltages: Iterable[float]) -> None:
        """Validate every target before performing any hardware operation."""
        if self._validate_voltage is None:
            return
        for voltage in voltages:
            self._validate_voltage(voltage)

    @contextmanager
    def _connection(self) -> Iterator[DCVoltageDevice]:
        """Yield a connected device and close on exit."""
        device = self._device_factory()
        try:
            yield device
        except BaseException:
            try:
                device.close()
            except BaseException:
                logger.exception(
                    "Failed to close DC voltage device connection while handling "
                    "an operation error."
                )
            raise
        else:
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
        profiles = {channel: profile for channel, (_, profile) in requests.items()}
        self._validate_voltages(voltage for voltage, _ in requests.values())
        self._validate_voltages(profile.idle_voltage_v for profile in profiles.values())
        try:
            self.apply_channels(requests)
            yield
        except BaseException:
            try:
                if profiles:
                    self.idle_channels(profiles)
            except BaseException:
                logger.exception(
                    "Failed to return DC voltage outputs to idle while handling "
                    "a measurement error."
                )
            raise
        else:
            if profiles:
                self.idle_channels(profiles)

    def _apply_voltage(
        self,
        device: DCVoltageDevice,
        *,
        channel: int,
        voltage: float,
        profile: DCVoltageProfile,
    ) -> None:
        """Ramp one enabled channel to a target voltage."""
        self._validate_voltages((voltage,))
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
        self._validate_voltages((profile.idle_voltage_v,))
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
        """Apply incremental setpoints and verify readback at the final target."""
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
        # Intermediate readback would add one hardware round trip per step.
        # Validate the final target instead so software ramps remain predictable.
        while abs(voltage - current) > step:
            current += direction * step
            device.set_voltage(channel, current)
            time.sleep(wait)
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
