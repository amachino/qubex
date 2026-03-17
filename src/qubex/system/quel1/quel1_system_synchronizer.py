"""Experiment-system synchronizer for QuEL-1 backend integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from qubex.core.parallel_executor import run_parallel, run_parallel_map
from qubex.system.control_system import PortType
from qubex.system.quel1.quel1_control_parameter_defaults import DEFAULT_CAPTURE_DELAY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController
    from qubex.system.control_system import Box, CapPort, GenPort
    from qubex.system.experiment_system import ExperimentSystem


class Quel1SystemSynchronizer:
    """Synchronize experiment-system models into QuEL-1 backend state."""

    def __init__(self, *, backend_controller: Quel1BackendController) -> None:
        self._backend_controller = backend_controller

    @property
    def backend_controller(self) -> Quel1BackendController:
        """Return backend controller bound to this synchronizer."""
        return self._backend_controller

    @property
    def supports_backend_settings_sync(self) -> bool:
        """Return whether QuEL-1 supports hardware snapshot synchronization."""
        return True

    @property
    def supports_mutable_backend_settings_cache(self) -> bool:
        """Return whether QuEL-1 supports mutable backend-settings cache writes."""
        return True

    def sync_experiment_system_to_backend_controller(
        self,
        experiment_system: ExperimentSystem,
    ) -> None:
        """Rebuild backend controller from one experiment-system model."""
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params
        clock_master_address = control_system.clock_master_address
        if len(control_system.boxes) > 1 and clock_master_address is None:
            raise ValueError(
                "Clock master address is required for multi-box QuEL-1 synchronization."
            )
        if clock_master_address is not None:
            self._backend_controller.define_clockmaster(
                ipaddr=clock_master_address,
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
                port_type = port.type.value
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
                if port_type in ("MNTR_OUT", "MNTR_IN"):
                    self._backend_controller.define_target(
                        target_name=port.id,
                        channel_name=port.channels[0].id,
                        target_frequency_ghz=0.0,
                    )

        for target in experiment_system.all_targets:
            self._backend_controller.define_target(
                target_name=target.label,
                channel_name=target.channel.id,
                target_frequency_ghz=target.frequency,
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
        experiment_system: ExperimentSystem,
        boxes: Sequence[Box],
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
    ) -> None:
        """Apply experiment-system port/channel parameters to hardware boxes."""
        del experiment_system, target_labels
        if parallel is None:
            parallel = True
        if not boxes:
            return
        if not parallel:
            for box in boxes:
                self.sync_box_to_hardware(box)
            return
        run_parallel(
            boxes,
            self.sync_box_to_hardware,
            on_error=self._log_box_sync_error,
        )

    def get_box_config_cache_snapshot(self) -> dict[str, dict]:
        """Return a snapshot of backend box-config cache when supported."""
        return dict(self._backend_controller.get_box_config_cache())

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Replace backend box-config cache when supported."""
        self._backend_controller.replace_box_config_cache(dict(box_configs))

    def fetch_backend_settings_from_hardware(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        """Fetch backend settings from hardware and skip failed box dumps."""
        if parallel is None:
            parallel = True
        boxes = [experiment_system.get_box(box_id) for box_id in box_ids]
        if not boxes:
            return {}

        def _dump_box(box: Box) -> dict[str, Any]:
            return self._backend_controller.dump_box(box.id)

        if not parallel:
            result: dict[str, dict] = {}
            for box in boxes:
                box_config = self._backend_controller.dump_box(box.id)
                if self._is_valid_dumped_box_config(box.id, box_config):
                    result[box.id] = box_config
            return result

        raw_result = run_parallel_map(
            boxes,
            _dump_box,
            key=lambda box: box.id,
            as_completed_order=True,
            on_error=self._fallback_dump_box_result,
        )
        return {
            box_id: box_config
            for box_id, box_config in raw_result.items()
            if self._is_valid_dumped_box_config(box_id, box_config)
        }

    def sync_backend_settings_to_backend_controller(
        self,
        *,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to backend controller cache."""
        self._backend_controller.update_box_config_cache(backend_settings)

    def sync_backend_settings_to_experiment_system(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: dict[str, dict],
    ) -> None:
        """Apply backend-settings snapshots to the in-memory experiment system."""
        updates: list[
            tuple[
                str,
                int,
                Literal["U", "L"] | None,
                int | None,
                int,
                list[int],
                int | None,
            ]
        ] = []
        for box_id, box_config in backend_settings.items():
            ports_config = box_config.get("ports", {})
            try:
                box = experiment_system.get_box(box_id)
            except KeyError:
                logger.warning("Box %s is not found.", box_id)
                continue
            for experiment_port in box.ports:
                if experiment_port.type in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                    continue
                port_number = experiment_port.number
                if not isinstance(port_number, int):
                    continue
                port_config = ports_config.get(port_number)
                if not isinstance(port_config, dict):
                    continue
                direction = port_config.get("direction")
                lo_freq_hz = port_config.get("lo_freq")
                lo_freq_hz = int(lo_freq_hz) if lo_freq_hz is not None else None
                cnco_freq_hz = int(port_config["cnco_freq"])
                if direction == "out":
                    raw_sideband = port_config.get("sideband")
                    sideband: Literal["U", "L"] | None = (
                        raw_sideband if raw_sideband in ("U", "L") else None
                    )
                    fullscale_current = port_config.get("fullscale_current")
                    fullscale_current = (
                        int(fullscale_current)
                        if fullscale_current is not None
                        else None
                    )
                    fnco_freqs_hz = [
                        int(channel["fnco_freq"])
                        for channel in port_config.get("channels", {}).values()
                    ]
                elif direction == "in":
                    sideband = None
                    fullscale_current = None
                    fnco_freqs_hz = [
                        int(channel["fnco_freq"])
                        for channel in port_config.get("runits", {}).values()
                    ]
                else:
                    continue
                channels = experiment_port.channels
                expected_fnco_count = len(channels) if len(channels) > 0 else None
                if (
                    expected_fnco_count is not None
                    and len(fnco_freqs_hz) != expected_fnco_count
                ):
                    logger.warning(
                        "Skipping backend port sync for %s:%s due to fnco count mismatch "
                        "(expected=%s, actual=%s).",
                        box_id,
                        port_number,
                        expected_fnco_count,
                        len(fnco_freqs_hz),
                    )
                    continue
                updates.append(
                    (
                        box_id,
                        port_number,
                        sideband,
                        lo_freq_hz,
                        cnco_freq_hz,
                        fnco_freqs_hz,
                        fullscale_current,
                    )
                )
        for (
            box_id,
            port_number,
            sideband,
            lo_freq_hz,
            cnco_freq_hz,
            fnco_freqs_hz,
            fullscale_current,
        ) in updates:
            experiment_system.control_system.set_port_params(
                box_id=box_id,
                port_number=port_number,
                sideband=sideband,
                lo_freq=lo_freq_hz,
                cnco_freq=cnco_freq_hz,
                fnco_freqs=fnco_freqs_hz,
                fullscale_current=fullscale_current,
            )

    def _sync_generator_port(self, *, box: Box, port: GenPort) -> None:
        """Apply one output-like port configuration."""
        try:
            self._backend_controller.config_port(
                box_name=box.id,
                port=port.number,
                lo_freq_hz=port.lo_freq,
                cnco_freq_hz=port.cnco_freq,
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
                    fnco_freq_hz=gen_channel.fnco_freq,
                )
        except Exception:
            logger.exception("Failed to configure %s", port.id)

    def _sync_capture_port(self, *, box: Box, port: CapPort) -> None:
        """Apply one input-like port configuration."""
        try:
            self._backend_controller.config_port(
                box_name=box.id,
                port=port.number,
                lo_freq_hz=port.lo_freq,
                cnco_freq_hz=port.cnco_freq,
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
                    fnco_freq_hz=cap_channel.fnco_freq,
                )
        except Exception:
            logger.exception("Failed to configure %s", port.id)

    @staticmethod
    def _log_box_sync_error(box: Box, exc: BaseException) -> None:
        """Log a failure during per-box hardware synchronization."""
        logger.exception("Failed to configure box %s", box.id, exc_info=exc)

    @staticmethod
    def _fallback_dump_box_result(box: Box, exc: BaseException) -> dict[str, Any]:
        """Log one dump failure and return an empty fallback payload."""
        logger.exception("Failed to dump box %s", box.id, exc_info=exc)
        return {}

    @staticmethod
    def _is_valid_dumped_box_config(box_id: str, box_config: Mapping[str, Any]) -> bool:
        """Return whether dumped box config is valid enough for sync application."""
        if not box_config:
            logger.warning("Skip empty backend settings for box %s.", box_id)
            return False
        ports = box_config.get("ports")
        if not isinstance(ports, Mapping):
            logger.warning(
                "Skip malformed backend settings for box %s: missing `ports` mapping.",
                box_id,
            )
            return False
        return True

    @staticmethod
    def _is_generator_port(port: GenPort | CapPort) -> TypeGuard[GenPort]:
        """Return whether one port should be configured through generator path."""
        return port.type.value in ("CTRL", "READ_OUT", "PUMP")

    @staticmethod
    def _is_capture_port(port: GenPort | CapPort) -> TypeGuard[CapPort]:
        """Return whether one port should be configured through capture path."""
        return port.type.value in ("READ_IN", "MNTR_IN")
