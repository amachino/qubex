"""Shared backend controller contracts and backend-family selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .backend_executor import BackendExecutionRequest, BackendExecutionResult
from .quel1 import Quel1BackendController
from .quel3 import Quel3BackendController

BackendKind = Literal["quel1", "quel3"]


@runtime_checkable
class BackendController(Protocol):
    """
    Measurement-facing backend controller contract.

    Required methods are the cross-layer execution/session operations used by
    measurement services. Link/clock operations remain capability-gated
    extensions.
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
    def box_config(self) -> dict[str, Any]:
        """Return connected box configuration cache."""
        ...

    def execute(self, *, request: BackendExecutionRequest) -> BackendExecutionResult:
        """Execute prepared backend request payload."""
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

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """Load skew calibration settings."""
        ...

    # Optional capabilities (backend dependent)
    def link_status(self, box_name: str) -> dict[int, bool]:
        """Return link status for one box."""
        ...

    def read_clocks(self, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """Read clock-related values for selected boxes."""
        ...

    def check_clocks(self, box_list: list[str]) -> bool:
        """Return whether clocks are synchronized."""
        ...

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> dict[str, Any]:
        """Link up selected boxes."""
        ...

    def relinkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Relink selected boxes."""
        ...

    def sync_clocks(self, box_list: list[str]) -> bool:
        """Synchronize clocks for selected boxes."""
        ...

    def resync_clocks(self, box_list: list[str]) -> bool:
        """Force re-synchronization of clocks for selected boxes."""
        ...


SystemBackendController: TypeAlias = Quel1BackendController | Quel3BackendController
