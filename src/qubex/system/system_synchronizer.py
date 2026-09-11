"""Protocol contracts for backend system synchronizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
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

    @property
    def supports_backend_settings_sync(self) -> bool:
        """Return whether hardware snapshot synchronization is supported."""
        ...

    @property
    def supports_mutable_backend_settings_cache(self) -> bool:
        """Return whether mutable backend-settings cache writes are supported."""
        ...

    def modified_capture_delay(
        self,
        *,
        experiment_system: ExperimentSystem,
        capture_delay: Mapping[int, int | float],
    ) -> AbstractContextManager[None]:
        """
        Convert nanosecond delays and temporarily apply backend-specific settings.

        Notes
        -----
        The manager validates mux keys and finite, non-negative numeric values.
        The synchronizer validates the backend resolution and owns temporary
        control parameters and controller state, restoring both on exit,
        including partial entry failures.
        """
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
        experiment_system: ExperimentSystem,
        boxes: Sequence[Box],
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
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

    def get_box_config_cache_snapshot(self) -> dict[str, dict]:
        """Return a snapshot of mutable backend cache state when supported."""
        ...

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Replace mutable backend cache state when supported."""
        ...
