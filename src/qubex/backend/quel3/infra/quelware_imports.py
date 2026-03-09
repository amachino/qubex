"""Shared helpers for importing quelware modules with workspace fallback."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, cast

from qubex.backend.quel3.interfaces import QuelwareClientFactory

Quel3ClientMode = Literal["server", "standalone"]
SUPPORTED_QUEL3_CLIENT_MODES: Final[frozenset[Quel3ClientMode]] = frozenset(
    {"server", "standalone"}
)


def _candidate_local_quelware_paths(root: Path) -> tuple[Path, ...]:
    """Return local quelware source directories checked for fallback import."""
    return (
        root / "lib" / "quelware-client-internal" / "quelware-client" / "src",
        root / "lib" / "quelware-client-internal" / "quelware-core" / "python" / "src",
        root / "packages" / "quelware-client" / "quelware-client" / "src",
        root / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
    )


def _append_local_quelware_paths() -> None:
    """Append local quelware source paths when present in the workspace."""
    root = Path(__file__).resolve().parents[5]
    for path in _candidate_local_quelware_paths(root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_module_with_workspace_fallback(module_name: str) -> ModuleType:
    """Import one module, retrying after local quelware path injection."""
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, SyntaxError):
        _append_local_quelware_paths()
        return importlib.import_module(module_name)


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
    client_mode: str,
    standalone_unit_label: str | None,
) -> QuelwareClientFactory:
    """Load one quelware client factory for the configured runtime mode."""
    normalized_client_mode = validate_quelware_client_runtime(
        client_mode=client_mode,
        standalone_unit_label=standalone_unit_label,
    )
    client_module = import_module_with_workspace_fallback("quelware_client.client")
    if normalized_client_mode == "server":
        return cast(QuelwareClientFactory, client_module.create_quelware_client)
    unit_label = standalone_unit_label
    return cast(
        QuelwareClientFactory,
        lambda endpoint, port: client_module.create_standalone_client(
            endpoint,
            port,
            unit_label=unit_label,
        ),
    )
