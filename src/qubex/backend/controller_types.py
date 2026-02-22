"""Measurement-facing backend controller contracts and capability protocols."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

from .backend_executor import BackendExecutionRequest, BackendExecutionResult

if TYPE_CHECKING:
    from .quel1 import Quel1BackendController
    from .quel3 import Quel3BackendController

BackendKind = Literal["quel1", "quel3"]


@runtime_checkable
class BackendController(Protocol):
    """
    Shared backend controller contract for measurement execution and sessions.

    This protocol defines the minimum API that both QuEL-1 and QuEL-3
    controllers must provide to the measurement layer:
    `hash`, `is_connected`, `sampling_period`, `execute`, `connect`,
    and `disconnect`.

    Backend-specific features are not part of this contract and are provided
    through optional capability protocols defined in this module.
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


@runtime_checkable
class BackendBoxConfigProvider(Protocol):
    """Capability protocol for reading connected box configuration cache."""

    @property
    def box_config(self) -> dict[str, Any]:
        """Return connected box configuration cache."""
        ...


@runtime_checkable
class BackendSkewYamlLoader(Protocol):
    """Capability protocol for loading skew calibration settings."""

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """Load skew calibration settings."""
        ...


@runtime_checkable
class BackendLinkStatusReader(Protocol):
    """Capability protocol for reading link status per box."""

    def link_status(self, box_name: str) -> dict[int, bool]:
        """Return link status for one box."""
        ...


@runtime_checkable
class BackendClockStatusReader(Protocol):
    """Capability protocol for reading and checking clocks."""

    def read_clocks(self, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """Read clock-related values for selected boxes."""
        ...

    def check_clocks(self, box_list: list[str]) -> bool:
        """Return whether clocks are synchronized."""
        ...


@runtime_checkable
class BackendLinkupOperator(Protocol):
    """Capability protocol for linkup operations."""

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> dict[str, Any]:
        """Link up selected boxes."""
        ...


@runtime_checkable
class BackendRelinkupOperator(Protocol):
    """Capability protocol for relinkup operations."""

    def relinkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """Relink selected boxes."""
        ...


@runtime_checkable
class BackendClockSynchronizer(Protocol):
    """Capability protocol for clock synchronization operations."""

    def sync_clocks(self, box_list: list[str]) -> bool:
        """Synchronize clocks for selected boxes."""
        ...


@runtime_checkable
class BackendClockResynchronizer(Protocol):
    """Capability protocol for clock re-synchronization operation."""

    def resync_clocks(self, box_list: list[str]) -> bool:
        """Force re-synchronization of clocks for selected boxes."""
        ...


SystemBackendController: TypeAlias = "Quel1BackendController | Quel3BackendController"
