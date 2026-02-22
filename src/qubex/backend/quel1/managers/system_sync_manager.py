# ruff: noqa: SLF001

"""Experiment-system synchronization manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeGuard

from qubex.backend.quel1.quel1_backend_constants import DEFAULT_CAPTURE_DELAY
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

from .configuration_manager import Quel1ConfigurationManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.control_system import Box, CapPort, GenPort
    from qubex.backend.experiment_system import ExperimentSystem


class Quel1SystemSyncManager:
    """Synchronize experiment-system models into QuEL-1 backend state."""

    def __init__(self, *, runtime_context: Quel1RuntimeContext) -> None:
        self._runtime_context = runtime_context
        self._configuration_manager = Quel1ConfigurationManager(
            runtime_context=runtime_context
        )

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Rebuild backend-controller topology from one experiment-system model."""
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params
        self._configuration_manager.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=True,
        )
        self._configuration_manager.set_box_options(
            {box.id: box.options for box in control_system.boxes}
        )

        for box in control_system.boxes:
            self._configuration_manager.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            for port in box.ports:
                port_type = getattr(port.type, "value", None)
                if port_type == "NA":
                    continue
                self._configuration_manager.define_port(
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
                    self._configuration_manager.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

                if port_type in ("PUMP", "MNTR_OUT", "MNTR_IN"):
                    self._configuration_manager.add_channel_target_relation(
                        channel_name=port.channels[0].id,
                        target_name=port.id,
                    )

        for target in experiment_system.all_targets:
            self._configuration_manager.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency=target.frequency,
            )

        self._configuration_manager.clear_command_queue()
        self.clear_cache()

    def sync_box_to_hardware(self, box: Box) -> None:
        """Apply one experiment-system box configuration to hardware."""
        for port in box.ports:
            if self._is_generator_port(port):
                self._sync_generator_port(box=box, port=port)
            elif self._is_capture_port(port):
                self._sync_capture_port(box=box, port=port)

    def clear_cache(self) -> None:
        """Clear cached box configuration data."""
        boxpool = self._runtime_context.boxpool_or_none()
        if boxpool is not None:
            boxpool._box_config_cache.clear()
        quel1system = self._runtime_context.quel1system_or_none()
        if quel1system is None:
            return
        quel1system.config_cache.clear()
        quel1system.config_fetched_at = None

    def replace_box_config_cache(
        self,
        box_configs: dict[str, dict[str, Any]],
    ) -> None:
        """Replace the full box-config cache snapshot."""
        boxpool = self._runtime_context.boxpool_or_none()
        if boxpool is None:
            if box_configs:
                raise ValueError("Boxes not connected. Call connect() method first.")
            return
        boxpool._box_config_cache = deepcopy(box_configs)
        self._replace_quel1system_box_cache(box_configs)

    def update_box_config_cache(
        self,
        box_configs: dict[str, dict[str, Any]],
    ) -> None:
        """Update box-config cache entries keyed by box name."""
        boxpool = self._runtime_context.boxpool_or_none()
        if boxpool is None:
            if box_configs:
                raise ValueError("Boxes not connected. Call connect() method first.")
            return
        for box_name, box_config in box_configs.items():
            boxpool._box_config_cache[box_name] = deepcopy(box_config)
        self._update_quel1system_box_cache(box_configs)

    def _sync_generator_port(self, *, box: Box, port: GenPort) -> None:
        """Apply one output-like port configuration."""
        try:
            self._configuration_manager.config_port(
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
                self._configuration_manager.config_channel(
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
            self._configuration_manager.config_port(
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
                self._configuration_manager.config_runit(
                    box_name=box.id,
                    port=port.number,
                    runit=cap_channel.number,
                    fnco_freq=cap_channel.fnco_freq,
                )
        except Exception:
            logger.exception("Failed to configure %s", port.id)

    def _replace_quel1system_box_cache(
        self,
        box_configs: dict[str, dict[str, Any]],
    ) -> None:
        """Replace the Quel1System-side box cache."""
        quel1system = self._runtime_context.quel1system_or_none()
        if quel1system is None:
            return
        quel1system.config_cache.clear()
        for box_name, box_config in box_configs.items():
            quel1system.config_cache[box_name] = deepcopy(box_config)
        quel1system.config_fetched_at = (
            datetime.now() if quel1system.config_cache else None
        )

    def _update_quel1system_box_cache(
        self,
        box_configs: dict[str, dict[str, Any]],
    ) -> None:
        """Update entries in the Quel1System-side box cache."""
        quel1system = self._runtime_context.quel1system_or_none()
        if quel1system is None:
            return
        for box_name, box_config in box_configs.items():
            quel1system.config_cache[box_name] = deepcopy(box_config)
        if quel1system.config_cache:
            quel1system.config_fetched_at = datetime.now()

    @staticmethod
    def _is_generator_port(port: GenPort | CapPort) -> TypeGuard[GenPort]:
        """Return whether one port should be configured through generator path."""
        return port.type.value in ("CTRL", "READ_OUT", "PUMP")

    @staticmethod
    def _is_capture_port(port: GenPort | CapPort) -> TypeGuard[CapPort]:
        """Return whether one port should be configured through capture path."""
        return port.type.value == "READ_IN"
