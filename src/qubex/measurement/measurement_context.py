"""Measurement context helpers backed by `SystemManager`."""

from __future__ import annotations

from collections.abc import Collection

from qubex.backend import (
    BackendController,
    ConfigLoader,
    ExperimentSystem,
    Mux,
    SystemManager,
)


class MeasurementContext:
    """Provide measurement context derived from selected qubits and system state."""

    def __init__(
        self,
        *,
        system_manager: SystemManager,
        qubits: Collection[str],
    ) -> None:
        self._system_manager = system_manager
        self._qubits = list(qubits)

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self._system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the active configuration loader."""
        return self._system_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment-system model."""
        return self._system_manager.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Return the active backend controller."""
        return self._system_manager.backend_controller

    @property
    def box_ids(self) -> list[str]:
        """Return box IDs corresponding to configured qubits."""
        boxes = self.experiment_system.get_boxes_for_qubits(self._qubits)
        return [box.id for box in boxes]

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Return mux map keyed by qubit label."""
        return {
            qubit: self.experiment_system.get_mux_by_qubit(qubit)
            for qubit in self._qubits
        }
