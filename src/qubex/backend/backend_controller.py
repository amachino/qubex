"""Measurement-facing backend controller contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from .quel1 import Quel1BackendController
    from .quel3 import Quel3BackendController

BackendKind = Literal["quel1", "quel3"]


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
    `hash`, `is_connected`, `sampling_period`, async `execute`, `connect`,
    and `disconnect`.
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
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        ...

    async def execute(
        self,
        *,
        request: BackendExecutionRequest,
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
