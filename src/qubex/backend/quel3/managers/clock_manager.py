"""Clock-related capability state for QuEL-3 backend controller."""

from __future__ import annotations

from qubex.backend.quel3.quel3_runtime_context import Quel3RuntimeContextReader


class Quel3ClockManager:
    """Expose QuEL-3 clock capability state."""

    def __init__(self, *, runtime_context: Quel3RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        return self._runtime_context.is_connected

    def ensure_supported(self, operation_name: str) -> None:
        """Raise an explicit error for unsupported clock operations."""
        raise NotImplementedError(
            f"QuEL-3 backend does not support `{operation_name}` clock operation."
        )
