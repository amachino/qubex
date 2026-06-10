"""Shared helpers for loading quelware runtime dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, cast

from qubex.backend.quel3.interfaces import QuelwareClientFactory

Quel3ClientMode = Literal["server"]
SUPPORTED_QUEL3_CLIENT_MODES: Final[frozenset[Quel3ClientMode]] = frozenset({"server"})


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
) -> Quel3ClientMode:
    """Validate one QuEL-3 client runtime and return normalized mode."""
    normalized_client_mode = normalize_quel3_client_mode(client_mode)
    if normalized_client_mode is None:
        raise ValueError(f"Unsupported QuEL-3 client mode: {client_mode!r}")
    return normalized_client_mode


def load_quelware_client_factory(
    *,
    client_mode: Quel3ClientMode,
    pat_path: str | None = None,
) -> QuelwareClientFactory:
    """Load one quelware client factory for the configured runtime mode."""
    validate_quelware_client_runtime(client_mode=client_mode)
    client_module = importlib.import_module("quelware_client.client")
    pat_provider: Callable[[], str] | None = None
    if pat_path is not None:
        path = Path(pat_path)
        pat_provider = lambda: path.read_text(encoding="utf-8").rstrip("\r\n")
    if pat_provider is not None:
        return cast(
            QuelwareClientFactory,
            lambda endpoint, port: client_module.create_quelware_client(
                endpoint,
                port,
                pat=pat_provider,
            ),
        )
    return cast(QuelwareClientFactory, client_module.create_quelware_client)
