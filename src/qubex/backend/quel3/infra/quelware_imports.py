"""Shared helpers for loading quelware runtime dependencies."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final, Literal, cast

from qubex.backend.quel3.interfaces import QuelwareClientFactory

Quel3ClientMode = Literal["server", "standalone"]
SUPPORTED_QUEL3_CLIENT_MODES: Final[frozenset[Quel3ClientMode]] = frozenset(
    {"server", "standalone"}
)
_STANDALONE_NOTICE_LOGGER: Final[str] = "quelware_client.client._standalone_grpc"
_STANDALONE_NOTICE_MESSAGE: Final[str] = (
    "NOTE: Standalone client is for testing purposes."
)


class _StandaloneNoticeFilter(logging.Filter):
    """Suppress only the repeated standalone-client testing notice."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == _STANDALONE_NOTICE_LOGGER
            and record.getMessage() == _STANDALONE_NOTICE_MESSAGE
        )


@contextmanager
def _suppress_standalone_notice() -> Iterator[None]:
    """Temporarily suppress the standalone-client testing notice."""
    logger = logging.getLogger(_STANDALONE_NOTICE_LOGGER)
    log_filter = _StandaloneNoticeFilter()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


def _create_standalone_client_safely(
    *,
    client_module: Any,
    endpoint: str,
    port: int,
    unit_label: str | None,
):
    """Create standalone client while suppressing the repeated testing notice."""
    with _suppress_standalone_notice():
        return client_module.create_standalone_client(
            endpoint,
            port,
            unit_label=unit_label,
        )


def normalize_quel3_client_mode(value: object) -> Quel3ClientMode | None:
    """Normalize one QuEL-3 client-mode value to the canonical literal."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_QUEL3_CLIENT_MODES:
            return cast(Quel3ClientMode, normalized)
    return None


def validate_quelware_client_runtime(
    *,
    client_mode: str,
    standalone_unit_label: str | None,
) -> Quel3ClientMode:
    """Validate one QuEL-3 client runtime combination and return normalized mode."""
    normalized_client_mode = normalize_quel3_client_mode(client_mode)
    if normalized_client_mode is None:
        raise ValueError(f"Unsupported QuEL-3 client mode: {client_mode!r}")
    if normalized_client_mode == "standalone" and standalone_unit_label is None:
        raise ValueError(
            "`standalone_unit_label` is required when `client_mode='standalone'`."
        )
    if normalized_client_mode == "server" and standalone_unit_label is not None:
        raise ValueError(
            "`standalone_unit_label` must be omitted when `client_mode='server'`."
        )
    return normalized_client_mode


def load_quelware_client_factory(
    *,
    client_mode: Quel3ClientMode,
    standalone_unit_label: str | None,
) -> QuelwareClientFactory:
    """Load one quelware client factory for the configured runtime mode."""
    normalized_client_mode = validate_quelware_client_runtime(
        client_mode=client_mode,
        standalone_unit_label=standalone_unit_label,
    )
    client_module = importlib.import_module("quelware_client.client")
    if normalized_client_mode == "server":
        return cast(QuelwareClientFactory, client_module.create_quelware_client)
    unit_label = standalone_unit_label
    return cast(
        QuelwareClientFactory,
        lambda endpoint, port: _create_standalone_client_safely(
            client_module=client_module,
            endpoint=endpoint,
            port=port,
            unit_label=unit_label,
        ),
    )
