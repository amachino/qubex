"""Shared runtime configuration for QuEL-3 managers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.interfaces import QuelwareClientFactory


@dataclass(frozen=True)
class Quel3RuntimeConfig:
    """Hold quelware runtime settings shared by QuEL-3 managers."""

    endpoint: str = "localhost"
    port: int = 50051
    client_mode: str = "server"
    pat_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize runtime settings after construction."""
        object.__setattr__(
            self,
            "client_mode",
            validate_quelware_client_runtime(client_mode=self.client_mode),
        )

    def load_client_factory(self) -> QuelwareClientFactory:
        """Load the quelware client factory for this runtime config."""
        return load_quelware_client_factory(
            client_mode=self.client_mode_value,
            pat_path=self.pat_path,
        )

    @property
    def client_mode_value(self) -> Quel3ClientMode:
        """Return client mode with the validated literal type."""
        return cast(Quel3ClientMode, self.client_mode)
