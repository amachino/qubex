"""Device and configuration operations for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Literal

from qubex.backend import (
    ConfigLoader,
    ExperimentSystem,
    Mux,
    SystemManager,
)
from qubex.backend.quel1 import DeviceController

logger = logging.getLogger(__name__)


class MeasurementBackendManager:
    """Handle config loading and hardware connectivity for measurement."""

    def __init__(
        self, *, system_manager: SystemManager, qubits: Collection[str]
    ) -> None:
        self._system_manager = system_manager
        self._qubits = list(qubits)

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self._system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the configuration loader."""
        return self._system_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the experiment system."""
        return self._system_manager.experiment_system

    @property
    def device_controller(self) -> DeviceController:
        """Return the device controller."""
        return self._system_manager.device_controller

    @cached_property
    def box_ids(self) -> list[str]:
        """Return box IDs corresponding to configured qubits."""
        boxes = self.experiment_system.get_boxes_for_qubits(self._qubits)
        return [box.id for box in boxes]

    @cached_property
    def mux_dict(self) -> dict[str, Mux]:
        """Return mux map keyed by qubit label."""
        return {
            qubit: self.experiment_system.get_mux_by_qubit(qubit)
            for qubit in self._qubits
        }

    def load(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ) -> None:
        """Load configuration and skew settings."""
        self.system_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )
        self.system_manager.load_skew_file(self.box_ids)

    def connect(self, *, sync_clocks: bool = True) -> None:
        """Connect to configured devices and optionally resync clocks."""
        if len(self.box_ids) == 0:
            logger.warning("No boxes are selected. Please check the configuration.")
            return
        self.device_controller.connect(self.box_ids)
        self.system_manager.pull(self.box_ids)
        if sync_clocks:
            self.device_controller.resync_clocks(self.box_ids)

    def is_connected(self) -> bool:
        """Return True if device controller is connected."""
        return self.device_controller.is_connected

    def check_link_status(self, box_list: list[str]) -> dict:
        """Check link status for the provided box list."""
        link_statuses = {
            box: self.device_controller.link_status(box) for box in box_list
        }
        is_linkedup = all(all(status.values()) for status in link_statuses.values())
        return {
            "status": is_linkedup,
            "links": link_statuses,
        }

    def check_clock_status(self, box_list: list[str]) -> dict:
        """Check clock synchronization status for the provided box list."""
        clocks = self.device_controller.read_clocks(box_list)
        clock_statuses = dict(
            zip(
                box_list,
                clocks,
                strict=True,
            )
        )
        is_synced = self.device_controller.check_clocks(box_list)
        return {
            "status": is_synced,
            "clocks": clock_statuses,
        }

    def linkup(self, box_list: list[str], noise_threshold: int | None = None) -> None:
        """Link up boxes and synchronize clocks."""
        self.device_controller.linkup_boxes(box_list, noise_threshold=noise_threshold)
        self.device_controller.sync_clocks(box_list)

    def relinkup(self, box_list: list[str]) -> None:
        """Relink up boxes and synchronize clocks."""
        self.device_controller.relinkup_boxes(box_list)
        self.device_controller.sync_clocks(box_list)

    @contextmanager
    def modified_frequencies(
        self, target_frequencies: dict[str, float]
    ) -> Iterator[None]:
        """Temporarily apply frequency overrides."""
        if target_frequencies is None:
            yield
        else:
            with self.system_manager.modified_frequencies(target_frequencies):
                yield
