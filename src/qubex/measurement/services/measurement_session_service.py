"""Session and connectivity services for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from qubex.backend import (
    BackendController,
    BackendKind,
    ConfigLoader,
    ExperimentSystem,
    SystemManager,
)
from qubex.measurement.measurement_context import MeasurementContext
from qubex.typing import ConfigurationMode

logger = logging.getLogger(__name__)


class MeasurementSessionService:
    """Handle config loading and hardware connectivity for measurement."""

    def __init__(
        self,
        *,
        system_manager: SystemManager,
        context: MeasurementContext,
    ) -> None:
        self._system_manager = system_manager
        self._context = context

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self._system_manager

    @property
    def context(self) -> MeasurementContext:
        """Return measurement context accessor."""
        return self._context

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the active configuration loader."""
        return self._context.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment-system model."""
        return self._context.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Return the active backend controller."""
        return self._context.backend_controller

    @property
    def box_ids(self) -> list[str]:
        """Return active box IDs for the selected qubits."""
        return self._context.box_ids

    def load(
        self,
        *,
        chip_id: str,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        configuration_mode: ConfigurationMode | None = None,
        backend_kind: BackendKind | None = None,
    ) -> None:
        """Load configuration and skew settings."""
        self.system_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            backend_kind=backend_kind,
        )
        self.load_skew_file()

    def load_skew_file(self) -> None:
        """Load skew calibration data from the current config directory."""
        skew_file_path = self.config_loader.config_path / "skew.yaml"
        if not skew_file_path.exists():
            logger.warning(f"Skew file not found: {skew_file_path}")
            return
        backend_controller = self.backend_controller
        load_skew_yaml = getattr(backend_controller, "load_skew_yaml", None)
        if not callable(load_skew_yaml):
            logger.info(
                "Skipping skew file load because this backend does not support skew calibration."
            )
            return
        try:
            load_skew_yaml(skew_file_path)
        except Exception:
            logger.exception("Failed to load the skew file.")

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect to configured devices and optionally resync clocks.

        Parameters
        ----------
        sync_clocks : bool | None, optional
            Whether to resync clocks, by default True.
        parallel : bool | None, optional
            Whether to use parallel backend connect path and fetch backend
            settings in parallel, by default True.
        """
        if sync_clocks is None:
            sync_clocks = True
        box_ids = self.box_ids
        if len(box_ids) == 0:
            logger.warning("No boxes are selected. Please check the configuration.")
            return
        backend_controller = self.backend_controller
        backend_controller.connect(box_ids, parallel=parallel)
        self.system_manager.pull(box_ids, parallel=parallel)
        if sync_clocks:
            resync_clocks = getattr(backend_controller, "resync_clocks", None)
            if callable(resync_clocks):
                resync_clocks(box_ids)
            else:
                logger.info(
                    "Skipping clock re-synchronization because this backend does not support it."
                )

    def is_connected(self) -> bool:
        """Return True if active backend controller is connected."""
        return self.backend_controller.is_connected

    def disconnect(self) -> None:
        """Disconnect backend resources held by the active controller."""
        self.backend_controller.disconnect()

    def check_link_status(self, box_list: list[str]) -> dict:
        """Check link status for the provided box list."""
        backend_controller = self.backend_controller
        link_status = cast(
            Callable[[str], dict[int, bool]] | None,
            getattr(backend_controller, "link_status", None),
        )
        if not callable(link_status):
            raise NotImplementedError(
                "Active backend does not support link status checks."
            )
        link_statuses = {box: link_status(box) for box in box_list}
        is_linkedup = all(all(status.values()) for status in link_statuses.values())
        return {
            "status": is_linkedup,
            "links": link_statuses,
        }

    def check_clock_status(self, box_list: list[str]) -> dict:
        """Check clock synchronization status for the provided box list."""
        backend_controller = self.backend_controller
        read_clocks = cast(
            Callable[[list[str]], list[tuple[bool, int, int]]] | None,
            getattr(backend_controller, "read_clocks", None),
        )
        check_clocks = cast(
            Callable[[list[str]], bool] | None,
            getattr(backend_controller, "check_clocks", None),
        )
        if not callable(read_clocks) or not callable(check_clocks):
            raise NotImplementedError(
                "Active backend does not support clock status checks."
            )
        clocks = read_clocks(box_list)
        clock_statuses = dict(
            zip(
                box_list,
                clocks,
                strict=True,
            )
        )
        is_synced = check_clocks(box_list)
        return {
            "status": is_synced,
            "clocks": clock_statuses,
        }

    def linkup(self, box_list: list[str], noise_threshold: int | None = None) -> None:
        """Link up boxes and synchronize clocks."""
        backend_controller = self.backend_controller
        linkup_boxes = cast(
            Callable[..., Any] | None,
            getattr(backend_controller, "linkup_boxes", None),
        )
        sync_clocks = cast(
            Callable[[list[str]], bool] | None,
            getattr(backend_controller, "sync_clocks", None),
        )
        if not callable(linkup_boxes):
            raise NotImplementedError("Active backend does not support linkup.")
        if not callable(sync_clocks):
            raise NotImplementedError(
                "Active backend does not support clock synchronization."
            )
        linkup_boxes(box_list, noise_threshold=noise_threshold)
        sync_clocks(box_list)

    def relinkup(self, box_list: list[str]) -> None:
        """Relink up boxes and synchronize clocks."""
        backend_controller = self.backend_controller
        relinkup_boxes = cast(
            Callable[[list[str]], None] | None,
            getattr(backend_controller, "relinkup_boxes", None),
        )
        sync_clocks = cast(
            Callable[[list[str]], bool] | None,
            getattr(backend_controller, "sync_clocks", None),
        )
        if not callable(relinkup_boxes):
            raise NotImplementedError("Active backend does not support relinkup.")
        if not callable(sync_clocks):
            raise NotImplementedError(
                "Active backend does not support clock synchronization."
            )
        relinkup_boxes(box_list)
        sync_clocks(box_list)

    @contextmanager
    def modified_frequencies(
        self,
        target_frequencies: dict[str, float],
    ) -> Iterator[None]:
        """Temporarily apply frequency overrides."""
        if target_frequencies is None:
            yield
        else:
            with self.system_manager.modified_frequencies(target_frequencies):
                yield
