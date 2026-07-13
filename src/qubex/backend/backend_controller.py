"""Measurement-facing backend controller contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .quel1 import Quel1BackendController
    from .quel3 import Quel3BackendController

BackendKind = Literal["quel1", "quel3"]
BackendExecutionMode = Literal["serial", "parallel"]
BACKEND_KIND_QUEL1: Final[BackendKind] = "quel1"
BACKEND_KIND_QUEL3: Final[BackendKind] = "quel3"
SUPPORTED_BACKEND_KINDS: Final[frozenset[BackendKind]] = frozenset(
    {BACKEND_KIND_QUEL1, BACKEND_KIND_QUEL3}
)


def normalize_backend_kind(value: object) -> BackendKind | None:
    """Normalize one backend-kind value to canonical lowercase literal."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_BACKEND_KINDS:
            return cast(BackendKind, normalized)
    return None


@dataclass(frozen=True)
class BackendExecutionRequest:
    """Backend-neutral execution request."""

    payload: Any


BackendExecutionResult: TypeAlias = Any


@runtime_checkable
class BackendController(Protocol):
    """
    Shared backend controller contract for measurement execution and sessions.

    This protocol defines the minimum API that both QuEL-1 and QuEL-3
    controllers must provide to the measurement layer:
    `hash`, `is_connected`, `sampling_period_ns`, `execute_sync`,
    `execute_async`, `connect`, and `disconnect`.
    """

    @property
    def hash(self) -> int:
        """Return a stable hash for controller state."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return whether backend resources are connected."""
        ...

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        ...

    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: BackendExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute prepared backend request payload synchronously."""
        ...

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: BackendExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute prepared backend request payload asynchronously."""
        ...

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Connect backend resources for selected boxes."""
        ...

    def disconnect(self) -> None:
        """Disconnect backend resources."""
        ...


SystemBackendController: TypeAlias = "Quel1BackendController | Quel3BackendController"
