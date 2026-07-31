"""External DC voltage controller configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .protocol import DCVoltageDeviceFactory


@dataclass(frozen=True)
class DCVoltageProfile:
    """Define resolved voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float = 0.1
    update_interval_s: float = 0.1
    safe_voltage_v: float = 0.0
    readback_tolerance_v: float = 1e-3


@dataclass(frozen=True)
class DCVoltageProfileOverride:
    """Override voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float | None = None
    update_interval_s: float | None = None
    safe_voltage_v: float | None = None
    readback_tolerance_v: float | None = None


@dataclass(frozen=True)
class DCVoltageControllerConfig:
    """Configure one external DC voltage controller."""

    driver: str = "ons61797"
    port: str | None = None
    ip_address: str | None = None
    max_set_attempts: int = 3
    device_factory: DCVoltageDeviceFactory | None = None
    voltage_defaults: DCVoltageProfile = field(
        default_factory=lambda: DCVoltageProfile(channel=1)
    )
    muxes: dict[int, DCVoltageProfileOverride] = field(default_factory=dict)

    def resolve_voltage_profile(self, mux_index: int) -> DCVoltageProfile:
        """Resolve the effective voltage-control profile for one mux."""
        override = self.muxes.get(
            mux_index,
            DCVoltageProfileOverride(channel=mux_index + 1),
        )
        values = {
            name: value
            for name in (
                "ramp_rate_v_per_s",
                "update_interval_s",
                "safe_voltage_v",
                "readback_tolerance_v",
            )
            if (value := getattr(override, name)) is not None
        }
        return replace(
            self.voltage_defaults,
            channel=override.channel,
            **values,
        )


@dataclass(frozen=True)
class ExternalDevicesConfig:
    """Configure external devices attached to one Qubex system."""

    dc_voltage_controllers: dict[str, DCVoltageControllerConfig] = field(
        default_factory=lambda: {"jpa_bias": DCVoltageControllerConfig()}
    )
