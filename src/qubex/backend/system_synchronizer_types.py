"""Protocol contracts for backend system synchronizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qubex.backend.control_system import Box
    from qubex.backend.controller_types import SystemBackendController
    from qubex.backend.experiment_system import ExperimentSystem


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
        """Rebuild backend-local topology from experiment-system state."""
        ...

    def sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """Apply experiment-system settings to hardware boxes."""
        ...

    def supports_box_settings_cache_sync(self) -> bool:
        """Return whether backend supports dump/cache synchronization APIs."""
        ...

    def supports_backend_settings_mutation(self) -> bool:
        """Return whether backend supports temporary backend-setting overrides."""
        ...

    def get_box_config_cache_snapshot(self) -> dict[str, dict]:
        """Return a snapshot of backend box-config cache when supported."""
        ...

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Replace backend box-config cache when supported."""
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

    def sync_backend_settings_to_device_controller(
        self,
        *,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to backend-controller cache."""
        ...

    def sync_backend_settings_to_experiment_system(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to in-memory experiment system."""
        ...
