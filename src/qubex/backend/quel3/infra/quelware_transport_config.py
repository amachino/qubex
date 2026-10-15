"""Typed configuration for QuEL-3 quelware transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

Quel3Transport = Literal["grpc", "http", "https"]
HttpScheme = Literal["http", "https"]

_SUPPORTED_TRANSPORTS = frozenset({"grpc", "http", "https"})


@dataclass(frozen=True)
class Quel3HttpTransportConfig:
    """Configure the protobuf-over-HTTP transport."""

    base_path: str = ""
    default_timeout_seconds: float | None = None
    proxy_url_path: str | None = None
    secret_header_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate file-backed HTTP transport options."""
        if self.proxy_url_path is not None and not self.proxy_url_path:
            raise ValueError("HTTP proxy URL path must not be empty")
        if "" in self.secret_header_paths:
            raise ValueError("Secret header names must not be empty")
        if "" in self.secret_header_paths.values():
            raise ValueError("Secret header paths must not be empty")

    def __hash__(self) -> int:
        """Return a hash including the configured secret header paths."""
        return hash(
            (
                self.base_path,
                self.default_timeout_seconds,
                self.proxy_url_path,
                frozenset(self.secret_header_paths.items()),
            )
        )


def validate_quel3_transport_config(
    *,
    transport: str,
    http_transport: Quel3HttpTransportConfig | None,
) -> Quel3Transport:
    """Normalize and validate one QuEL-3 transport configuration."""
    normalized_transport = transport.strip().lower()
    if normalized_transport not in _SUPPORTED_TRANSPORTS:
        raise ValueError(f"Unsupported QuEL-3 transport: {transport!r}")

    if normalized_transport == "grpc" and http_transport is not None:
        raise ValueError("http_transport cannot be configured for gRPC transport")

    if normalized_transport == "http" and http_transport is not None:
        if http_transport.secret_header_paths:
            raise ValueError(
                "Secret headers can be configured only for HTTPS transport"
            )

    return cast(Quel3Transport, normalized_transport)


__all__ = [
    "HttpScheme",
    "Quel3HttpTransportConfig",
    "Quel3Transport",
    "validate_quel3_transport_config",
]
