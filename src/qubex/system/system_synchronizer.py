"""Protocol contracts for backend system synchronizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qubex.backend.backend_controller import SystemBackendController
    from qubex.system.control_system import Box
    from qubex.system.experiment_system import ExperimentSystem


class SystemSynchronizer(Protocol):
    """Backend-specific synchronizer interface consumed by `SystemManager`."""

    @property
    def backend_controller(self) -> SystemBackendController:
        """Return backend controller bound to this synchronizer."""
        ...

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Rebuild backend controller state from experiment-system state."""
        ...

    def sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """Apply experiment-system settings to hardware boxes."""
        ...

    def fetch_backend_settings_from_hardware(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch raw backend settings from hardware for selected boxes."""
        ...

    def sync_backend_settings_to_backend_controller(
        self,
        *,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to backend controller cache."""
        ...

    def sync_backend_settings_to_experiment_system(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to in-memory experiment system."""
        ...
