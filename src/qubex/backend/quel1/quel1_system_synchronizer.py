"""Experiment-system synchronizer for QuEL-1 backend integration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeGuard

from qubex.backend.parallel_box_executor import run_parallel_each
from qubex.backend.quel1.quel1_backend_constants import DEFAULT_CAPTURE_DELAY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.control_system import Box, CapPort, GenPort
    from qubex.backend.experiment_system import ExperimentSystem
    from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


class Quel1SystemSynchronizer:
    """Synchronize experiment-system models into QuEL-1 backend state."""

    def __init__(self, *, backend_controller: Quel1BackendController) -> None:
        self._backend_controller = backend_controller

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Rebuild backend-controller topology from one experiment-system model."""
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params
        self._backend_controller.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=True,
        )
        self._backend_controller.set_box_options(
            {box.id: box.options for box in control_system.boxes}
        )

        for box in control_system.boxes:
            self._backend_controller.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            for port in box.ports:
                port_type = getattr(port.type, "value", None)
                if port_type == "NA":
                    continue
                self._backend_controller.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,
                )

                for channel in port.channels:
                    if port_type == "READ_IN":
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            continue
                        ndelay_or_nwait = control_params.get_capture_delay(mux.index)
                    elif port_type == "MNTR_IN":
                        ndelay_or_nwait = DEFAULT_CAPTURE_DELAY
                    else:
                        ndelay_or_nwait = 0
                    self._backend_controller.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

                if port_type in ("PUMP", "MNTR_OUT", "MNTR_IN"):
                    self._backend_controller.add_channel_target_relation(
                        channel_name=port.channels[0].id,
                        target_name=port.id,
                    )

        for target in experiment_system.all_targets:
            self._backend_controller.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency=target.frequency,
            )

        self._backend_controller.clear_command_queue()
        self._backend_controller.clear_cache()

    def sync_box_to_hardware(self, box: Box) -> None:
        """Apply one experiment-system box configuration to hardware."""
        for port in box.ports:
            if self._is_generator_port(port):
                self._sync_generator_port(box=box, port=port)
            elif self._is_capture_port(port):
                self._sync_capture_port(box=box, port=port)

    def sync_experiment_system_to_hardware(
        self,
        *,
        boxes: Sequence[Box],
        parallel: bool | None = None,
    ) -> None:
        """Apply experiment-system port/channel parameters to hardware boxes."""
        if parallel is None:
            parallel = True
        if not boxes:
            return
        if not parallel:
            for box in boxes:
                self.sync_box_to_hardware(box)
            return
        run_parallel_each(
            boxes,
            self.sync_box_to_hardware,
            on_error=self._log_box_sync_error,
        )

    def _sync_generator_port(self, *, box: Box, port: GenPort) -> None:
        """Apply one output-like port configuration."""
        try:
            self._backend_controller.config_port(
                box_name=box.id,
                port=port.number,
                lo_freq=port.lo_freq,
                cnco_freq=port.cnco_freq,
                vatt=port.vatt,
                sideband=port.sideband,
                fullscale_current=port.fullscale_current,
                rfswitch=port.rfswitch,
            )
            for gen_channel in port.channels:
                self._backend_controller.config_channel(
                    box_name=box.id,
                    port=port.number,
                    channel=gen_channel.number,
                    fnco_freq=gen_channel.fnco_freq,
                )
        except Exception:
            logger.exception("Failed to configure %s", port.id)

    def _sync_capture_port(self, *, box: Box, port: CapPort) -> None:
        """Apply one input-like port configuration."""
        try:
            self._backend_controller.config_port(
                box_name=box.id,
                port=port.number,
                lo_freq=port.lo_freq,
                cnco_freq=port.cnco_freq,
                vatt=None,
                sideband=None,
                fullscale_current=None,
                rfswitch=port.rfswitch,
            )
            for cap_channel in port.channels:
                self._backend_controller.config_runit(
                    box_name=box.id,
                    port=port.number,
                    runit=cap_channel.number,
                    fnco_freq=cap_channel.fnco_freq,
                )
        except Exception:
            logger.exception("Failed to configure %s", port.id)

    @staticmethod
    def _log_box_sync_error(box: Box, exc: BaseException) -> None:
        """Log a failure during per-box hardware synchronization."""
        logger.exception("Failed to configure box %s", box.id, exc_info=exc)

    @staticmethod
    def _is_generator_port(port: GenPort | CapPort) -> TypeGuard[GenPort]:
        """Return whether one port should be configured through generator path."""
        return port.type.value in ("CTRL", "READ_OUT", "PUMP")

    @staticmethod
    def _is_capture_port(port: GenPort | CapPort) -> TypeGuard[CapPort]:
        """Return whether one port should be configured through capture path."""
        return port.type.value == "READ_IN"
