"""System synchronizer for QuEL-3 backend integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from qubex.system.control_system import GenPort
from qubex.system.target_type import TargetType

from .quel3_target_deploy_planner import Quel3TargetDeployPlanner

if TYPE_CHECKING:
    from qubex.backend.quel3.models import Quel3InstrumentState, Quel3ResourceSnapshot
    from qubex.backend.quel3.quel3_backend_controller import Quel3BackendController
    from qubex.system.control_system import Box, GenChannel
    from qubex.system.experiment_system import ExperimentSystem
    from qubex.system.target import Target


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

    def resolve_custom_target_channel(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_id: str,
        port_number: int,
        channel_number: int | None,
    ) -> GenChannel:
        """Resolve the logical channel for a custom target."""
        port = experiment_system.control_system.get_port(box_id, port_number)
        if not isinstance(port, GenPort):
            raise TypeError(
                f"Custom target registration port `{port.id}` must be a GenPort."
            )
        if channel_number is None:
            channel_number = 0
        if channel_number < 0 or channel_number >= len(port.channels):
            raise IndexError(
                f"Channel number `{channel_number}` is out of range for port `{port.id}`."
            )
        return port.channels[channel_number]

    def register_custom_target(
        self,
        *,
        experiment_system: ExperimentSystem,
        target: Target,
        update_backend_settings: bool = False,
        push_boxes: Callable[[Sequence[str]], None],
    ) -> None:
        """Deploy and read back the instrument unconditionally before publishing its target."""
        del update_backend_settings, push_boxes
        if (
            not target.label
            or target.label != target.label.strip()
            or ":" in target.label
        ):
            raise ValueError(
                "Instrument alias must be nonempty with no ':' or surrounding whitespace."
            )
        port = target.channel.port
        previous = experiment_system.gen_targets.get(target.label)
        if previous is not None:
            previous_port = previous.channel.port
            if (previous_port.box_id, previous_port.number) != (
                port.box_id,
                port.number,
            ):
                raise ValueError(
                    f"Target `{target.label}` is already registered on another port."
                )
        if target.type == TargetType.READ:
            mux = experiment_system.get_mux_by_readout_port(port)
            expected_mux = experiment_system.get_mux_by_qubit(target.qubit)
            if mux is None or mux.index != expected_mux.index:
                raise ValueError("Readout port must belong to the target qubit's mux.")
            inputs = [
                cap_port
                for read_mux, cap_port in experiment_system.wiring_info.read_in
                if read_mux.index == mux.index
            ]
            if len(inputs) != 1 or inputs[0].box_id != port.box_id:
                raise ValueError(
                    "Readout port requires exactly one read-in pair on the same unit."
                )
        specification = self._deploy_planner.build_instrument_spec(
            experiment_system=experiment_system,
            target=target,
        )
        self._backend_controller.deploy_instrument(
            instrument=specification, append=True
        )
        experiment_system.add_target(target)

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
        snapshot = self._backend_controller.get_resource_snapshot(
            unit_labels=tuple(box_ids),
            parallel=True if parallel is None else parallel,
            level="instrument",
        )
        return self._project_backend_settings(snapshot=snapshot, unit_labels=box_ids)

    @classmethod
    def _project_backend_settings(
        cls,
        *,
        snapshot: Quel3ResourceSnapshot,
        unit_labels: Sequence[str],
    ) -> dict[str, dict]:
        """Project resource snapshot into backend settings keyed by selected unit labels."""
        settings: dict[str, dict] = {
            unit_label: {"instruments": {}} for unit_label in unit_labels
        }

        for instrument in snapshot.instruments:
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


def create_system_synchronizer(
    backend_controller: Quel3BackendController,
) -> Quel3SystemSynchronizer:
    """Create a QuEL-3 synchronizer bound to the supplied controller."""
    return Quel3SystemSynchronizer(backend_controller=backend_controller)
