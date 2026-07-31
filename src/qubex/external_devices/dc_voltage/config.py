"""External DC voltage controller configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TypeVar

from .protocol import DCVoltageDeviceFactory


@dataclass(frozen=True)
class DCVoltageProfile:
    """Define resolved voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float = 0.1
    update_interval_s: float = 0.1
    safe_voltage_v: float = 0.0
    readback_tolerance_v: float = 1e-3
    max_set_attempts: int = 3


@dataclass(frozen=True)
class DCVoltageProfileOverride:
    """Override voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float | None = None
    update_interval_s: float | None = None
    safe_voltage_v: float | None = None
    readback_tolerance_v: float | None = None
    max_set_attempts: int | None = None


@dataclass(frozen=True)
class DCVoltageControllerConfig:
    """Configure one external DC voltage controller."""

    driver: str = "ons61797"
    connection: dict[str, object] = field(default_factory=dict)
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
                "max_set_attempts",
            )
            if (value := getattr(override, name)) is not None
        }
        return replace(
            self.voltage_defaults,
            channel=override.channel,
            **values,
        )

    @classmethod
    def from_dict(cls, raw_config: object) -> DCVoltageControllerConfig:
        """Create a normalized controller config from one YAML mapping."""
        if not isinstance(raw_config, Mapping):
            raise TypeError("DC voltage controller config must be a mapping.")
        driver = raw_config.get("driver", "ons61797")
        if not isinstance(driver, str):
            raise TypeError("DC voltage controller `driver` must be a string.")
        connection = raw_config.get("connection", {})
        if not isinstance(connection, Mapping):
            raise TypeError("DC voltage controller `connection` must be a mapping.")
        voltage_control = _nested_mapping(raw_config, "voltage_control")
        defaults = _nested_mapping(voltage_control, "defaults")
        default_profile = _parse_voltage_profile(
            defaults,
            channel=1,
            base=None,
        )
        muxes = _nested_mapping(voltage_control, "muxes")
        normalized_muxes: dict[int, DCVoltageProfileOverride] = {}
        for mux_index, values in muxes.items():
            if type(mux_index) is not int:
                raise TypeError("`voltage_control.muxes` must use integer mux indices.")
            if mux_index < 0:
                raise ValueError("DC voltage mux indices must be non-negative.")
            if not isinstance(values, Mapping):
                raise TypeError("Each DC voltage mux profile must be a mapping.")
            channel = values.get("channel")
            if type(channel) is not int or channel < 1:
                raise ValueError("DC voltage channels must be positive integers.")
            resolved = _parse_voltage_profile(
                values,
                channel=channel,
                base=default_profile,
            )
            normalized_muxes[mux_index] = DCVoltageProfileOverride(
                channel=channel,
                ramp_rate_v_per_s=_override_value(
                    resolved.ramp_rate_v_per_s,
                    default_profile.ramp_rate_v_per_s,
                ),
                update_interval_s=_override_value(
                    resolved.update_interval_s,
                    default_profile.update_interval_s,
                ),
                safe_voltage_v=_override_value(
                    resolved.safe_voltage_v,
                    default_profile.safe_voltage_v,
                ),
                readback_tolerance_v=_override_value(
                    resolved.readback_tolerance_v,
                    default_profile.readback_tolerance_v,
                ),
                max_set_attempts=_override_value(
                    resolved.max_set_attempts,
                    default_profile.max_set_attempts,
                ),
            )
        channels = [profile.channel for profile in normalized_muxes.values()]
        if len(set(channels)) != len(channels):
            raise ValueError(
                "`voltage_control.muxes` must not contain duplicate channels."
            )
        return cls(
            driver=driver,
            connection=dict(connection),
            voltage_defaults=default_profile,
            muxes=normalized_muxes,
        )


@dataclass(frozen=True)
class ExternalDevicesConfig:
    """Configure external devices attached to one Qubex system."""

    dc_voltage_controllers: dict[str, DCVoltageControllerConfig] = field(
        default_factory=lambda: {"jpa_bias": DCVoltageControllerConfig()}
    )

    @classmethod
    def from_dict(cls, raw_config: object) -> ExternalDevicesConfig:
        """Create external-device configuration from one YAML mapping."""
        if raw_config is None or raw_config == {}:
            return cls()
        if not isinstance(raw_config, Mapping):
            raise TypeError("`external_devices` must be a mapping.")
        controllers = raw_config.get("dc_voltage_controllers", {})
        if not isinstance(controllers, Mapping):
            raise TypeError("`dc_voltage_controllers` must be a mapping.")
        normalized: dict[str, DCVoltageControllerConfig] = {}
        for name, value in controllers.items():
            if not isinstance(name, str):
                raise TypeError("DC voltage controller names must be strings.")
            normalized[name] = DCVoltageControllerConfig.from_dict(value)
        return cls(dc_voltage_controllers=normalized)


def _parse_voltage_profile(
    values: Mapping[object, object],
    *,
    channel: int,
    base: DCVoltageProfile | None,
) -> DCVoltageProfile:
    """Parse one complete voltage profile with optional inherited values."""
    ramp = _nested_mapping(values, "ramp")
    shutdown = _nested_mapping(values, "shutdown")
    readback = _nested_mapping(values, "readback")
    profile = DCVoltageProfile(
        channel=channel,
        ramp_rate_v_per_s=_float_value(
            ramp,
            "rate_v_per_s",
            default=base.ramp_rate_v_per_s if base else 0.1,
        ),
        update_interval_s=_float_value(
            ramp,
            "step_interval_s",
            default=base.update_interval_s if base else 0.1,
        ),
        safe_voltage_v=_float_value(
            shutdown,
            "voltage_v",
            default=base.safe_voltage_v if base else 0.0,
        ),
        readback_tolerance_v=_float_value(
            readback,
            "tolerance_v",
            default=base.readback_tolerance_v if base else 1e-3,
        ),
        max_set_attempts=_int_value(
            readback,
            "max_attempts",
            default=base.max_set_attempts if base else 3,
        ),
    )
    _validate_voltage_profile(profile)
    return profile


def _nested_mapping(
    values: Mapping[object, object],
    key: str,
) -> Mapping[object, object]:
    """Return one optional nested config mapping."""
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"`{key}` must be a mapping.")
    return value


def _optional_float(
    values: Mapping[object, object],
    key: str,
) -> float | None:
    """Return an optional numeric config value as a float."""
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{key}` must be numeric.")
    return float(value)


def _float_value(
    values: Mapping[object, object],
    key: str,
    *,
    default: float,
) -> float:
    """Return one numeric config value or its default."""
    value = _optional_float(values, key)
    return default if value is None else value


def _int_value(
    values: Mapping[object, object],
    key: str,
    *,
    default: int,
) -> int:
    """Return one integer config value or its default."""
    value = values.get(key)
    if value is None:
        return default
    if type(value) is not int:
        raise TypeError(f"`{key}` must be an integer.")
    return value


def _validate_voltage_profile(profile: DCVoltageProfile) -> None:
    """Validate one resolved DC voltage profile."""
    if profile.ramp_rate_v_per_s <= 0:
        raise ValueError("`rate_v_per_s` must be positive.")
    if profile.update_interval_s <= 0:
        raise ValueError("`step_interval_s` must be positive.")
    if profile.readback_tolerance_v < 0:
        raise ValueError("`tolerance_v` must be non-negative.")
    if profile.max_set_attempts < 1:
        raise ValueError("`max_attempts` must be a positive integer.")


_T = TypeVar("_T")


def _override_value(value: _T, default: _T) -> _T | None:
    """Return only values that differ from the inherited default."""
    return None if value == default else value
