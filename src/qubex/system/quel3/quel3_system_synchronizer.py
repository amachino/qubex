"""System synchronizer for QuEL-3 backend integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qubex.backend.quel3.quel3_backend_controller import Quel3BackendController
    from qubex.system.control_system import Box
    from qubex.system.experiment_system import ExperimentSystem


class Quel3SystemSynchronizer:
    """No-op synchronizer for QuEL-3 until sync APIs are supported."""

    def __init__(self, *, backend_controller: Quel3BackendController) -> None:
        self._backend_controller = backend_controller

    @property
    def backend_controller(self) -> Quel3BackendController:
        """Return backend controller bound to this synchronizer."""
        return self._backend_controller

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """No-op: QuEL-3 backend controller model sync is not implemented."""
        del experiment_system

    def sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """No-op: QuEL-3 hardware sync path is not implemented."""
        del boxes, parallel

    def fetch_backend_settings_from_hardware(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch raw backend settings from hardware for selected boxes."""
        del experiment_system, box_ids, parallel
        return {}

    def sync_backend_settings_to_backend_controller(
        self,
        *,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to backend controller cache."""
        del backend_settings

    def sync_backend_settings_to_experiment_system(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to the in-memory experiment system."""
        del experiment_system, backend_settings
