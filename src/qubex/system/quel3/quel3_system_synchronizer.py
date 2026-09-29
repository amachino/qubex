"""System synchronizer for QuEL-3 backend integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .quel3_target_deploy_planner import Quel3TargetDeployPlanner

if TYPE_CHECKING:
    from qubex.backend.quel3.models import Quel3HardwareState, Quel3InstrumentState
    from qubex.backend.quel3.quel3_backend_controller import Quel3BackendController
    from qubex.system.control_system import Box
    from qubex.system.experiment_system import ExperimentSystem


class Quel3SystemSynchronizer:
    """Synchronize QuEL-3 logical targets to deployed instruments."""

    def __init__(
        self,
        *,
        backend_controller: Quel3BackendController,
        deploy_planner: Quel3TargetDeployPlanner | None = None,
    ) -> None:
        self._backend_controller = backend_controller
        self._deploy_planner = (
            deploy_planner if deploy_planner is not None else Quel3TargetDeployPlanner()
        )

    @property
    def backend_controller(self) -> Quel3BackendController:
        """Return backend controller bound to this synchronizer."""
        return self._backend_controller

    @property
    def deploy_planner(self) -> Quel3TargetDeployPlanner:
        """Return QuEL-3 deploy planner used for push-time configuration."""
        return self._deploy_planner

    @property
    def supports_backend_settings_sync(self) -> bool:
        """Return whether QuEL-3 supports hardware snapshot synchronization."""
        return True

    @property
    def supports_mutable_backend_settings_cache(self) -> bool:
        """Return whether QuEL-3 supports mutable backend-settings cache writes."""
        return False

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """No-op: QuEL-3 does not rebuild controller state from `ExperimentSystem`."""
        del experiment_system

    def sync_experiment_system_to_hardware(
        self,
        *,
        experiment_system: ExperimentSystem,
        boxes: Sequence[Box],
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
    ) -> None:
        """Deploy instruments for selected boxes from the current target registry."""
        box_ids = [box.id for box in boxes]
        if len(box_ids) == 0:
            return
        configuration = self._deploy_planner.build_configuration(
            experiment_system=experiment_system,
            box_ids=box_ids,
            target_labels=target_labels,
        )
        # Treat QuEL-3 instrument deploy as the push-time hardware sync
        # equivalent of QuEL-1 CNCO/FNCO updates. Execution paths should only
        # resolve and use the instruments that push configured.
        self._backend_controller.deploy_instruments(
            configuration=configuration,
            parallel=True if parallel is None else parallel,
        )

    def fetch_backend_settings_from_hardware(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch normalized instrument snapshots from hardware for selected boxes."""
        del experiment_system
        if not box_ids:
            return {}
        state = self._backend_controller.hardware_state_reader.collect_state(
            unit_labels=tuple(box_ids),
            include_diagnostics=False,
            parallel=True if parallel is None else parallel,
            view="instruments",
        )
        return self._project_backend_settings(state=state, unit_labels=box_ids)

    @classmethod
    def _project_backend_settings(
        cls,
        *,
        state: Quel3HardwareState,
        unit_labels: Sequence[str],
    ) -> dict[str, dict]:
        """Project hardware state into backend settings keyed by selected unit labels."""
        settings: dict[str, dict] = {
            unit_label: {"instruments": {}} for unit_label in unit_labels
        }

        for instrument in state.instruments:
            alias = instrument.normalized_alias or instrument.alias
            if alias is None:
                continue
            if instrument.unit_label in settings:
                settings[instrument.unit_label]["instruments"][alias] = (
                    cls._backend_settings_instrument(instrument)
                )
        return settings

    @staticmethod
    def _backend_settings_instrument(instrument: Quel3InstrumentState) -> dict:
        """Return backend-settings data for one instrument state."""
        definition: dict[str, object] = {
            "alias": instrument.alias or instrument.normalized_alias or "",
            "role": instrument.role,
        }
        if instrument.mode is not None:
            definition["mode"] = instrument.mode
        profile: dict[str, float] = {}
        if instrument.frequency_range_min_hz is not None:
            profile["frequency_range_min"] = instrument.frequency_range_min_hz
        if instrument.frequency_range_max_hz is not None:
            profile["frequency_range_max"] = instrument.frequency_range_max_hz
        if profile:
            definition["profile"] = profile
        return {
            "resource_id": instrument.id,
            "port_id": instrument.port_id,
            "role": instrument.role,
            "definition": definition,
        }

    def sync_backend_settings_to_backend_controller(
        self,
        *,
        backend_settings: dict[str, dict],
    ) -> None:
        """Leave live instrument state unchanged when applying saved settings."""
        del backend_settings

    def sync_backend_settings_to_experiment_system(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to the in-memory experiment system."""
        del experiment_system, backend_settings

    def get_box_config_cache_snapshot(self) -> dict[str, dict]:
        """Return empty cache snapshot because QuEL-3 cache mutation is unsupported."""
        return {}

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Ignore cache replacement because QuEL-3 cache mutation is unsupported."""
        del box_configs
