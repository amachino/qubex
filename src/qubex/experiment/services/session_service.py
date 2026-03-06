"""Session lifecycle service for Experiment context operations."""

from __future__ import annotations

import logging
from collections.abc import Collection

from qubex.experiment.experiment_context import ExperimentContext
from qubex.measurement import Measurement
from qubex.pulse import set_sampling_period
from qubex.typing import ConfigurationMode

logger = logging.getLogger(__name__)


class SessionService:
    """
    Handle session and configuration lifecycle for `ExperimentContext`.

    Ownership policy:
    - Measurement session operations delegate to `Measurement`.
    - Configuration synchronization delegates to `SystemManager`.
    """

    def __init__(
        self,
        *,
        experiment_context: ExperimentContext,
    ) -> None:
        self._ctx = experiment_context
        self._sync_pulse_sampling_period()

    @property
    def ctx(self) -> ExperimentContext:
        """Return backing experiment context."""
        return self._ctx

    @property
    def measurement(self) -> Measurement:
        """Return backing measurement facade."""
        return self.ctx.measurement

    def _sync_pulse_sampling_period(self) -> float:
        """Synchronize pulse-library sampling period with measurement dt."""
        sampling_period = float(self.measurement.sampling_period)
        set_sampling_period(sampling_period)
        return sampling_period

    def is_connected(self) -> bool:
        """Return whether measurement backend is connected."""
        return self.measurement.is_connected()

    def disconnect(self) -> None:
        """Disconnect backend resources via measurement facade."""
        self.measurement.disconnect()

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        """Connect measurement backend and synchronize sampling period."""
        try:
            self.measurement.connect(
                sync_clocks=sync_clocks,
                parallel=parallel,
            )
            self._sync_pulse_sampling_period()
            logger.info("Successfully connected.")
        except Exception:
            logger.exception("Failed to connect to the devices.")
            raise

    def reload(self) -> None:
        """Reload measurement configuration and synchronize sampling period."""
        try:
            self.measurement.reload(configuration_mode=self.ctx.configuration_mode)
            self._sync_pulse_sampling_period()
            logger.info("Successfully reloaded.")
        except Exception:
            logger.exception("Failed to reload the devices.")
            raise

    def check_status(self) -> None:
        """Log connectivity, clock, and configuration status."""
        if not self.is_connected():
            logger.warning(
                "Not connected to the devices. Call `connect()` method first."
            )
            return

        box_ids = self.ctx.box_ids
        if len(box_ids) == 0:
            logger.warning("No boxes are selected.")
            return

        measurement = self.measurement

        link_status = measurement.check_link_status(box_ids)
        if link_status["status"]:
            logger.info("Link status: OK")
        else:
            logger.warning("Link status: NG")
        logger.info(link_status["links"])

        clock_status = measurement.check_clock_status(box_ids)
        if clock_status["status"]:
            logger.info("Clock status: OK")
        else:
            logger.warning("Clock status: NG")
        logger.info(clock_status["clocks"])

        system_manager = self.ctx.system_manager
        config_status = system_manager.is_synced(box_ids=box_ids)
        if config_status:
            logger.info("Config status: OK")
        else:
            logger.warning("Config status: NG")
        logger.info(system_manager.backend_settings)

    def linkup(
        self,
        box_ids: list[str] | None = None,
        noise_threshold: int | None = None,
    ) -> None:
        """Link up boxes through measurement lifecycle path."""
        selected_box_ids = box_ids if box_ids is not None else self.ctx.box_ids
        self.measurement.linkup(
            selected_box_ids,
            noise_threshold=noise_threshold,
        )

    def resync_clocks(
        self,
        box_ids: list[str] | None = None,
    ) -> None:
        """Resynchronize clocks through backend optional capability."""
        selected_box_ids = box_ids if box_ids is not None else self.ctx.box_ids
        backend_controller = self.ctx.backend_controller
        resync_clocks = getattr(backend_controller, "resync_clocks", None)
        if not callable(resync_clocks):
            raise NotImplementedError(
                "Active backend does not support clock re-synchronization."
            )
        resync_clocks(selected_box_ids)

    def configure(
        self,
        *,
        box_ids: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        mode: ConfigurationMode | None = None,
    ) -> None:
        """Reload configuration through SystemManager and push to boxes."""
        if isinstance(box_ids, str):
            box_ids = [box_ids]
        if isinstance(exclude, str):
            exclude = [exclude]
        if mode is None:
            mode = self.ctx.configuration_mode

        system_manager = self.ctx.system_manager
        system_manager.load(
            chip_id=self.ctx.chip_id,
            config_dir=self.ctx.config_path,
            params_dir=self.ctx.params_path,
            targets_to_exclude=exclude,
            configuration_mode=mode,
        )
        resolved_box_ids = box_ids or self.ctx.box_ids
        system_manager.push(
            box_ids=resolved_box_ids,
            target_labels=list(self.ctx.targets),
        )
        self._sync_pulse_sampling_period()

    def reset_awg_and_capunits(
        self,
        *,
        box_ids: str | Collection[str] | None = None,
        qubits: Collection[str] | None = None,
    ) -> None:
        """Reset AWG and CAP units using backend optional capability."""
        resolved_box_ids: list[str] = []
        if qubits is not None:
            boxes = self.ctx.experiment_system.get_boxes_for_qubits(qubits)
            resolved_box_ids += [box.id for box in boxes]
        if len(resolved_box_ids) == 0:
            if isinstance(box_ids, str):
                resolved_box_ids = [box_ids]
            elif box_ids is not None:
                resolved_box_ids = list(box_ids)
            else:
                resolved_box_ids = self.ctx.box_ids

        backend_controller = self.ctx.backend_controller
        initialize_awg_and_capunits = getattr(
            backend_controller, "initialize_awg_and_capunits", None
        )
        if not callable(initialize_awg_and_capunits):
            raise NotImplementedError(
                "Active backend does not support AWG/CAP unit reset."
            )
        initialize_awg_and_capunits(resolved_box_ids)
