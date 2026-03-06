"""System synchronizer for QuEL-3 backend integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .quel3_configuration_manager import Quel3ConfigurationManager

if TYPE_CHECKING:
    from qubex.backend.quel3.quel3_backend_controller import Quel3BackendController
    from qubex.system.control_system import Box
    from qubex.system.experiment_system import ExperimentSystem


class Quel3SystemSynchronizer:
    """Synchronize QuEL-3 logical targets to deployed instruments."""

    def __init__(
        self,
        *,
        backend_controller: Quel3BackendController,
        configuration_manager: Quel3ConfigurationManager | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._configuration_manager = (
            configuration_manager
            if configuration_manager is not None
            else Quel3ConfigurationManager(
                quelware_endpoint=backend_controller.quelware_endpoint,
                quelware_port=backend_controller.quelware_port,
            )
        )
        self._experiment_system: ExperimentSystem | None = None

    @property
    def backend_controller(self) -> Quel3BackendController:
        """Return backend controller bound to this synchronizer."""
        return self._backend_controller

    @property
    def configuration_manager(self) -> Quel3ConfigurationManager:
        """Return QuEL-3 push-time configuration manager."""
        return self._configuration_manager

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Store current experiment-system model for upcoming push operation."""
        self._experiment_system = experiment_system

    def sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """Deploy instruments for selected boxes from the current target registry."""
        del parallel
        if self._experiment_system is None:
            raise RuntimeError(
                "Experiment system is not synchronized for QuEL-3 push. "
                "Call load() before push()."
            )
        box_ids = [box.id for box in boxes]
        if len(box_ids) == 0:
            return
        self._configuration_manager.deploy_instruments_from_target_registry(
            experiment_system=self._experiment_system,
            box_ids=box_ids,
        )

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
