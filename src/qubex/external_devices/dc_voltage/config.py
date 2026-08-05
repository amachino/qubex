"""External DC voltage controller configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeVar

from .protocol import DCVoltageDeviceFactory


@dataclass(frozen=True)
class DCVoltageDeviceConfig:
    """Configure one external DC voltage source device."""

    driver: str
    params: dict[str, object] = field(default_factory=dict)
    channels: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, device_id: str, raw_config: object) -> DCVoltageDeviceConfig:
        """Create one normalized device config from one YAML mapping."""
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"Device {device_id!r} config must be a mapping.")
        unknown = set(raw_config) - {"driver", "channels", "params"}
        if unknown:
            raise ValueError(
                f"Unknown device {device_id!r} settings: {sorted(unknown)}. "
                "Put driver-specific settings under `params`."
            )
        driver = raw_config.get("driver")
        if not isinstance(driver, str) or not driver.strip():
            raise ValueError(
                f"Device {device_id!r} requires a non-empty `driver` string."
            )
        raw_params = raw_config.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise TypeError(f"Device {device_id!r} `params` must be a mapping.")
        params = {str(key): value for key, value in raw_params.items()}
        raw_channels = raw_config.get("channels")
        channels: tuple[int, ...] | None = None
        if raw_channels is not None:
            if not isinstance(raw_channels, (list, tuple)) or not raw_channels:
                raise ValueError(
                    f"Device {device_id!r} `channels` must be a non-empty list."
                )
            for channel in raw_channels:
                if type(channel) is not int or channel < 1:
                    raise ValueError(
                        f"Device {device_id!r} channels must be positive integers."
                    )
            if len(set(raw_channels)) != len(raw_channels):
                raise ValueError(
                    f"Device {device_id!r} `channels` must not contain duplicates."
                )
            channels = tuple(raw_channels)
        return cls(
            driver=driver,
            params=params,
            channels=channels,
        )


def parse_device_output_ref(ref: object) -> tuple[str, int]:
    """Parse one `DEVICE-CHANNEL` output reference into its device and channel."""
    if not isinstance(ref, str):
        raise TypeError("DC voltage `bias` must be a `DEVICE-CHANNEL` string.")
    device_id, separator, channel_text = ref.rpartition("-")
    if not separator or not device_id or not channel_text.isdigit():
        raise ValueError(
            f"DC voltage bias {ref!r} must use the `DEVICE-CHANNEL` form, "
            "e.g. `Qblox1-15`."
        )
    channel = int(channel_text)
    if channel < 1:
        raise ValueError(f"DC voltage bias {ref!r} channel must be positive.")
    return device_id, channel


@dataclass(frozen=True)
class DCVoltageProfile:
    """Define resolved voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float = 0.1
    ramp_step_size_v: float = 0.01
    ramp_wait_s: float = 0.1
    reset_voltage_v: float = 0.0
    idle_voltage_v: float = 0.0
    readback_tolerance_v: float = 1e-3
    max_set_attempts: int = 3


@dataclass(frozen=True)
class DCVoltageProfileOverride:
    """Override voltage-control settings for one mux."""

    channel: int
    ramp_rate_v_per_s: float | None = None
    ramp_step_size_v: float | None = None
    ramp_wait_s: float | None = None
    reset_voltage_v: float | None = None
    readback_tolerance_v: float | None = None
    max_set_attempts: int | None = None


@dataclass(frozen=True)
class DCVoltageControllerConfig:
    """Configure one external DC voltage controller."""

    driver: str = "ons61797"
    params: dict[str, object] = field(default_factory=dict)
    device_id: str | None = None
    channels: tuple[int, ...] | None = None
    device_factory: DCVoltageDeviceFactory | None = None
    voltage_defaults: DCVoltageProfile = field(
        default_factory=lambda: DCVoltageProfile(channel=1)
    )
    muxes: dict[int, DCVoltageProfileOverride] = field(default_factory=dict)
    role: str = "bias"

    def resolve_voltage_profile(self, mux_index: int) -> DCVoltageProfile:
        """Resolve the explicitly configured voltage-control profile for one mux."""
        override = self.muxes.get(mux_index)
        if override is None:
            raise ValueError(
                f"Mux {mux_index} has no DC voltage wiring configured. "
                "Add an entry to the `wiring` list in `external_devices.yaml`."
            )
        values = {
            name: value
            for name in (
                "ramp_rate_v_per_s",
                "ramp_step_size_v",
                "ramp_wait_s",
                "reset_voltage_v",
                "readback_tolerance_v",
                "max_set_attempts",
            )
            if (value := getattr(override, name)) is not None
        }
        resolved = replace(
            self.voltage_defaults,
            channel=override.channel,
            **values,
        )
        return replace(resolved, idle_voltage_v=resolved.reset_voltage_v)

    @classmethod
    def from_dict(
        cls,
        raw_config: object,
        *,
        devices: Mapping[str, DCVoltageDeviceConfig],
        wiring: Mapping[str, Mapping[int, tuple[str, int]]],
    ) -> DCVoltageControllerConfig:
        """Create a normalized controller config from one YAML mapping."""
        if not isinstance(raw_config, Mapping):
            raise TypeError("DC voltage settings entry must be a mapping.")
        unknown = set(raw_config) - {
            "role",
            "ramp",
            "readback",
            "reset_voltage",
            "overrides",
        }
        if unknown:
            raise ValueError(f"Unknown DC voltage settings: {sorted(unknown)}.")
        role = raw_config.get("role", "bias")
        if not isinstance(role, str) or not role.strip():
            raise TypeError("DC voltage `role` must be a non-empty string.")
        default_profile = _parse_voltage_profile(
            raw_config,
            channel=1,
            base=None,
        )
        role_wiring = wiring.get(role)
        if not role_wiring:
            raise ValueError(
                f"No `wiring` entry defines a {role!r} output for this "
                "DC voltage controller."
            )
        device_ids = {device_id for device_id, _ in role_wiring.values()}
        if len(device_ids) > 1:
            raise ValueError(
                f"All {role!r} outputs of one DC voltage controller must "
                f"reference the same device, found {sorted(device_ids)}."
            )
        overrides = _parse_profile_overrides(raw_config, set(role_wiring))
        normalized_muxes: dict[int, DCVoltageProfileOverride] = {}
        for mux_index, (_, channel) in role_wiring.items():
            resolved = _parse_voltage_profile(
                overrides.get(mux_index, {}),
                channel=channel,
                base=default_profile,
            )
            normalized_muxes[mux_index] = DCVoltageProfileOverride(
                channel=channel,
                ramp_rate_v_per_s=_override_value(
                    resolved.ramp_rate_v_per_s,
                    default_profile.ramp_rate_v_per_s,
                ),
                ramp_step_size_v=_override_value(
                    resolved.ramp_step_size_v,
                    default_profile.ramp_step_size_v,
                ),
                ramp_wait_s=_override_value(
                    resolved.ramp_wait_s,
                    default_profile.ramp_wait_s,
                ),
                reset_voltage_v=_override_value(
                    resolved.reset_voltage_v,
                    default_profile.reset_voltage_v,
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
        device_id = next(iter(device_ids))
        device = devices[device_id]
        return cls(
            driver=device.driver,
            role=role,
            params=dict(device.params),
            device_id=device_id,
            channels=device.channels,
            voltage_defaults=default_profile,
            muxes=normalized_muxes,
        )


def _parse_wiring(
    raw_wiring: object,
    devices: Mapping[str, DCVoltageDeviceConfig],
) -> dict[str, dict[int, tuple[str, int]]]:
    """Parse the wiring list into role -> mux -> (device, channel)."""
    if not isinstance(raw_wiring, Sequence) or isinstance(raw_wiring, (str, bytes)):
        raise TypeError(
            "`wiring` must be a list of entries, e.g. `- mux: 8` + `bias: Qblox1-15`."
        )
    roles: dict[str, dict[int, tuple[str, int]]] = {}
    used_outputs: set[tuple[str, int]] = set()
    seen_muxes: set[int] = set()
    for entry in raw_wiring:
        if not isinstance(entry, Mapping):
            raise TypeError("Each `wiring` entry must be a mapping.")
        mux_index = entry.get("mux")
        if type(mux_index) is not int:
            raise TypeError("Each `wiring` entry requires an integer `mux`.")
        if mux_index < 0:
            raise ValueError("`wiring` mux indices must be non-negative.")
        if mux_index in seen_muxes:
            raise ValueError(f"`wiring` lists mux {mux_index} twice.")
        seen_muxes.add(mux_index)
        outputs = {key: value for key, value in entry.items() if key != "mux"}
        if not outputs:
            raise ValueError(f"`wiring` entry for mux {mux_index} defines no outputs.")
        for role, ref in outputs.items():
            if not isinstance(role, str) or not role.strip():
                raise TypeError("`wiring` output roles must be non-empty strings.")
            device_id, channel = parse_device_output_ref(ref)
            device = devices.get(device_id)
            if device is None:
                raise ValueError(
                    f"`wiring` mux {mux_index} {role} references unknown "
                    f"device {device_id!r}. Define it in `devices`."
                )
            if device.channels is not None and channel not in device.channels:
                raise ValueError(
                    f"`wiring` mux {mux_index} {role} references channel "
                    f"{channel}, which is not in device {device_id!r} "
                    "`channels`."
                )
            if (device_id, channel) in used_outputs:
                raise ValueError(
                    f"`wiring` connects device output {ref!r} more than once."
                )
            used_outputs.add((device_id, channel))
            roles.setdefault(role, {})[mux_index] = (device_id, channel)
    return roles


def _parse_profile_overrides(
    settings: Mapping[object, object],
    wired_muxes: set[int],
) -> dict[int, Mapping[object, object]]:
    """Parse per-mux voltage-control overrides for wired muxes."""
    raw_overrides = settings.get("overrides", [])
    if not isinstance(raw_overrides, Sequence) or isinstance(
        raw_overrides, (str, bytes)
    ):
        raise TypeError("`overrides` must be a list of entries.")
    overrides: dict[int, Mapping[object, object]] = {}
    for entry in raw_overrides:
        if not isinstance(entry, Mapping):
            raise TypeError("Each `overrides` entry must be a mapping.")
        unknown = set(entry) - {
            "mux",
            "ramp",
            "readback",
            "reset_voltage",
        }
        if unknown:
            raise ValueError(f"Unknown `overrides` settings: {sorted(unknown)}.")
        mux_index = entry.get("mux")
        if type(mux_index) is not int:
            raise TypeError("Each `overrides` entry requires an integer `mux`.")
        if mux_index in overrides:
            raise ValueError(f"`overrides` lists mux {mux_index} twice.")
        if mux_index not in wired_muxes:
            raise ValueError(f"`overrides` mux {mux_index} has no `wiring` entry.")
        overrides[mux_index] = entry
    return overrides


@dataclass(frozen=True)
class ExternalDevicesConfig:
    """Configure external devices attached to one Qubex system."""

    devices: dict[str, DCVoltageDeviceConfig] = field(default_factory=dict)
    wiring: dict[str, dict[int, tuple[str, int]]] = field(default_factory=dict)
    dc_voltage: DCVoltageControllerConfig = field(
        default_factory=DCVoltageControllerConfig
    )

    @classmethod
    def from_dict(cls, raw_config: object) -> ExternalDevicesConfig:
        """Create external-device configuration from one YAML mapping."""
        if raw_config is None or raw_config == {}:
            return cls()
        if not isinstance(raw_config, Mapping):
            raise TypeError("`external_devices` must be a mapping.")
        unknown = set(raw_config) - {"devices", "wiring", "settings"}
        if unknown:
            raise ValueError(f"Unknown external-device settings: {sorted(unknown)}.")
        raw_devices = raw_config.get("devices", {})
        if not isinstance(raw_devices, Mapping):
            raise TypeError("`devices` must be a mapping.")
        devices: dict[str, DCVoltageDeviceConfig] = {}
        for device_id, value in raw_devices.items():
            if not isinstance(device_id, str) or not device_id.strip():
                raise TypeError("Device names must be non-empty strings.")
            devices[device_id] = DCVoltageDeviceConfig.from_dict(device_id, value)
        wiring = _parse_wiring(raw_config.get("wiring", []), devices)
        settings = raw_config.get("settings", {})
        if not isinstance(settings, Mapping):
            raise TypeError("`settings` must be a mapping.")
        if not wiring and not settings:
            return cls(devices=devices)
        dc_voltage = DCVoltageControllerConfig.from_dict(
            settings,
            devices=devices,
            wiring=wiring,
        )
        unused_roles = sorted(set(wiring) - {dc_voltage.role})
        if unused_roles:
            raise ValueError(
                f"Unused DC voltage wiring roles: {unused_roles}. The configured "
                f"controller uses role {dc_voltage.role!r}."
            )
        return cls(
            devices=devices,
            wiring=wiring,
            dc_voltage=dc_voltage,
        )


def _parse_voltage_profile(
    values: Mapping[object, object],
    *,
    channel: int,
    base: DCVoltageProfile | None,
) -> DCVoltageProfile:
    """Parse one complete voltage profile with optional inherited values."""
    ramp = _nested_mapping(
        values,
        "ramp",
        allowed_keys={"rate_v_per_s", "step_size_v", "wait_s"},
    )
    readback = _nested_mapping(
        values,
        "readback",
        allowed_keys={"tolerance_v", "max_attempts"},
    )
    profile = DCVoltageProfile(
        channel=channel,
        ramp_rate_v_per_s=_float_value(
            ramp,
            "rate_v_per_s",
            default=base.ramp_rate_v_per_s if base else 0.1,
        ),
        ramp_step_size_v=_float_value(
            ramp,
            "step_size_v",
            default=base.ramp_step_size_v if base else 0.01,
        ),
        ramp_wait_s=_float_value(
            ramp,
            "wait_s",
            default=base.ramp_wait_s if base else 0.1,
        ),
        reset_voltage_v=_float_value(
            values,
            "reset_voltage",
            default=base.reset_voltage_v if base else 0.0,
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
    profile = replace(profile, idle_voltage_v=profile.reset_voltage_v)
    _validate_voltage_profile(profile)
    return profile


def _nested_mapping(
    values: Mapping[object, object],
    key: str,
    *,
    allowed_keys: set[str],
) -> Mapping[object, object]:
    """Return one optional nested config mapping with known keys only."""
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"`{key}` must be a mapping.")
    unknown = set(value) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown `{key}` settings: {sorted(unknown)}.")
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
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"`{key}` must be finite.")
    return resolved


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
    if profile.ramp_step_size_v <= 0:
        raise ValueError("`step_size_v` must be positive.")
    if profile.ramp_wait_s <= 0:
        raise ValueError("`wait_s` must be positive.")
    if profile.readback_tolerance_v < 0:
        raise ValueError("`tolerance_v` must be non-negative.")
    if profile.max_set_attempts < 1:
        raise ValueError("`max_attempts` must be a positive integer.")


_T = TypeVar("_T")


def _override_value(value: _T, default: _T) -> _T | None:
    """Return only values that differ from the inherited default."""
    return None if value == default else value
