# ruff: noqa: SLF001

"""Configuration and box-dump manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        Quel1BoxCommonProtocol as Quel1Box,
        SequencerProtocol as Sequencer,
    )


class Quel1ConfigurationManager:
    """Handle configuration, define, and dump operations for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContext) -> None:
        self._runtime_context = runtime_context

    def set_box_options(self, box_options: dict[str, tuple[str, ...]]) -> None:
        """Set per-box relink option labels."""
        self._runtime_context.set_box_options(box_options)

    def dump_box(self, *, box_name: str) -> dict:
        """Dump one box configuration and tolerate box-level errors."""
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
            )
            return box.dump_box()
        except Exception:
            logger.exception(f"Failed to dump box {box_name}.")
            return {}

    def dump_port(
        self,
        *,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> dict:
        """Dump one port configuration and tolerate box-level errors."""
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
            )
            return box.dump_port(port_number)
        except Exception:
            logger.exception(f"Failed to dump port {port_number} of box {box_name}.")
            return {}

    def config_port(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        lo_freq: float | None,
        cnco_freq: float | None,
        vatt: int | None,
        sideband: str | None,
        fullscale_current: int | None,
        rfswitch: str | None,
    ) -> None:
        """Configure one box port."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        if box.boxtype == "quel1se-riken8":
            vatt = None
            sideband = None
        if box.boxtype == "quel1se-riken8" and port not in box.get_input_ports():
            lo_freq = None
        box.config_port(
            port=port,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )

    def config_channel(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq: float | None,
    ) -> None:
        """Configure one box channel."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        box.config_channel(
            port=port,
            channel=channel,
            fnco_freq=fnco_freq,
        )

    def config_runit(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None,
    ) -> None:
        """Configure one box runit."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        box.config_runit(
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
        )

    def define_clockmaster(
        self,
        *,
        ipaddr: str,
        reset: bool = True,
    ) -> None:
        """Define clockmaster in qubecalib."""
        self._runtime_context.qubecalib.define_clockmaster(
            ipaddr=ipaddr,
            reset=reset,
        )

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """Define one box in qubecalib."""
        self._runtime_context.qubecalib.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
        )

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> None:
        """Define one port in qubecalib."""
        self._runtime_context.qubecalib.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
        )

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """Define one channel in qubecalib."""
        self._runtime_context.qubecalib.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(
        self,
        *,
        channel_name: str,
        target_name: str,
    ) -> None:
        """Add one channel-target relation if missing."""
        relation = (channel_name, target_name)
        sysdb = self._runtime_context.qubecalib.sysdb
        if relation not in sysdb._relation_channel_target:
            sysdb._relation_channel_target.append(relation)

    def define_target(
        self,
        *,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ) -> None:
        """Define one target in qubecalib."""
        self._runtime_context.qubecalib.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    def modify_target_frequency(
        self,
        *,
        target: str,
        frequency: float,
    ) -> None:
        """Modify one target frequency in qubecalib."""
        self._runtime_context.qubecalib.modify_target_frequency(
            target,
            frequency,
        )

    def modify_target_frequencies(
        self,
        *,
        frequencies: dict[str, float],
    ) -> None:
        """Modify multiple target frequencies in qubecalib."""
        for target, frequency in frequencies.items():
            self.modify_target_frequency(
                target=target,
                frequency=frequency,
            )

    def add_sequencer(self, *, sequencer: Sequencer) -> None:
        """Add one sequencer into qubecalib executor queue."""
        self._runtime_context.qubecalib._executor.add_command(sequencer)

    def show_command_queue(self) -> str:
        """Return current qubecalib command queue string."""
        return self._runtime_context.qubecalib.show_command_queue()

    def clear_command_queue(self) -> None:
        """Clear qubecalib command queue."""
        self._runtime_context.qubecalib.clear_command_queue()

    def _resolve_box(
        self,
        *,
        box_name: str,
        reconnect: bool,
    ) -> Quel1Box:
        """Resolve a box from runtime context or create it lazily."""
        self._runtime_context.validate_box_availability(box_name)
        boxpool = self._runtime_context.boxpool_or_none()
        if boxpool is not None and box_name in boxpool._boxes:
            return boxpool._boxes[box_name][0]
        db = self._runtime_context.qubecalib.system_config_database
        return db.create_box(box_name, reconnect=reconnect)
