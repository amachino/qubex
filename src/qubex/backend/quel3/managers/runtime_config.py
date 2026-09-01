"""Shared runtime configuration for QuEL-3 managers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.infra.quelware_transport_config import (
    Quel3HttpTransportConfig,
    Quel3Transport,
    validate_quel3_transport_config,
)
from qubex.backend.quel3.interfaces import QuelwareClientFactory


@dataclass(frozen=True)
class Quel3RuntimeConfig:
    """Hold quelware runtime settings shared by QuEL-3 managers."""

    endpoint: str = "localhost"
    port: int | None = None
    client_mode: str = "server"
    pat_path: str | None = None
    transport: str = "grpc"
    http_transport: Quel3HttpTransportConfig | None = None

    def __post_init__(self) -> None:
        """Normalize runtime settings after construction."""
        object.__setattr__(
            self,
            "client_mode",
            validate_quelware_client_runtime(client_mode=self.client_mode),
        )
        object.__setattr__(
            self,
            "transport",
            validate_quel3_transport_config(
                transport=self.transport,
                http_transport=self.http_transport,
            ),
        )
        if self.port is None and self.transport == "grpc":
            object.__setattr__(self, "port", 50051)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> Quel3RuntimeConfig:
        """Create runtime settings from one QuEL-3 system configuration."""
        config = value or {}
        if "standalone_unit_label" in config:
            raise ValueError(
                "QuEL-3 standalone runtime config `standalone_unit_label` is no longer supported."
            )
        endpoint = _get_config_value(config, "quelware_endpoint", "endpoint")
        port = _get_config_value(config, "quelware_port", "port")
        pat_path = _get_config_value(config, "quelware_pat_path", "pat_path")
        client_mode = config.get("client_mode")
        transport = config.get("transport")
        return cls(
            endpoint="localhost" if endpoint is None else cast(str, endpoint),
            port=None if port is None else cast(int, port),
            client_mode=("server" if client_mode is None else cast(str, client_mode)),
            pat_path=None if pat_path is None else cast(str, pat_path),
            transport="grpc" if transport is None else cast(str, transport),
            http_transport=_parse_http_transport_config(config.get("http_transport")),
        )

    def load_client_factory(self) -> QuelwareClientFactory:
        """Load the quelware client factory for this runtime config."""
        return load_quelware_client_factory(
            client_mode=self.client_mode_value,
            pat_path=self.pat_path,
            transport=self.transport_value,
            http_transport=self.http_transport,
        )

    @property
    def client_mode_value(self) -> Quel3ClientMode:
        """Return client mode with the validated literal type."""
        return cast(Quel3ClientMode, self.client_mode)

    @property
    def transport_value(self) -> Quel3Transport:
        """Return transport with the validated literal type."""
        return cast(Quel3Transport, self.transport)


__all__ = [
    "Quel3HttpTransportConfig",
    "Quel3RuntimeConfig",
]


def _get_config_value(
    config: Mapping[str, object],
    *keys: str,
) -> object | None:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _parse_http_transport_config(value: object) -> Quel3HttpTransportConfig | None:
    if value is None:
        return None
    if isinstance(value, Quel3HttpTransportConfig):
        return value
    config = _require_mapping(value, "http_transport")
    _reject_unknown_keys(
        config,
        {
            "base_path",
            "default_timeout_seconds",
            "proxy",
            "secret_header_paths",
        },
        "http_transport",
    )

    base_path = config.get("base_path", "")
    if not isinstance(base_path, str):
        raise TypeError("`http_transport.base_path` must be a string.")
    default_timeout_seconds = config.get("default_timeout_seconds")
    if isinstance(default_timeout_seconds, bool) or (
        default_timeout_seconds is not None
        and not isinstance(default_timeout_seconds, (int, float))
    ):
        raise TypeError(
            "`http_transport.default_timeout_seconds` must be a number or null."
        )

    proxy = _optional_nested_mapping(config.get("proxy"), "http_transport.proxy")
    proxy_url_path = (
        None
        if proxy is None
        else _required_string(proxy, "url_path", "http_transport.proxy")
    )
    if proxy is not None:
        _reject_unknown_keys(proxy, {"url_path"}, "http_transport.proxy")

    secret_header_paths_value = config.get("secret_header_paths", {})
    secret_header_paths_config = _require_mapping(
        secret_header_paths_value,
        "http_transport.secret_header_paths",
    )
    secret_header_paths: dict[str, str] = {}
    for name, path in secret_header_paths_config.items():
        if not isinstance(name, str) or not isinstance(path, str):
            raise TypeError(
                "`http_transport.secret_header_paths` must map strings to strings."
            )
        secret_header_paths[name] = path

    return Quel3HttpTransportConfig(
        base_path=base_path,
        default_timeout_seconds=(
            None if default_timeout_seconds is None else float(default_timeout_seconds)
        ),
        proxy_url_path=proxy_url_path,
        secret_header_paths=secret_header_paths,
    )


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{path}` must be a mapping.")
    return cast(Mapping[str, object], value)


def _optional_nested_mapping(
    value: object,
    path: str,
) -> Mapping[str, object] | None:
    return None if value is None else _require_mapping(value, path)


def _reject_unknown_keys(
    config: Mapping[str, object],
    supported_keys: set[str],
    path: str,
) -> None:
    unknown_keys = set(config) - supported_keys
    if unknown_keys:
        formatted_keys = ", ".join(sorted(map(str, unknown_keys)))
        raise ValueError(f"Unsupported `{path}` keys: {formatted_keys}")


def _required_string(
    config: Mapping[str, object],
    key: str,
    path: str,
) -> str:
    if key not in config:
        raise ValueError(f"`{path}.{key}` is required.")
    value = config[key]
    if not isinstance(value, str):
        raise TypeError(f"`{path}.{key}` must be a string.")
    return value


def _optional_string(
    config: Mapping[str, object],
    key: str,
    path: str,
) -> str | None:
    value = config.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"`{path}.{key}` must be a string.")
    return cast(str | None, value)
