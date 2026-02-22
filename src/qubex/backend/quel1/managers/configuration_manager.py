# ruff: noqa: SLF001

"""Configuration and box-dump manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.quel1_qubecalib_protocols import (
        QubeCalibProtocol as QubeCalib,
        Quel1BoxCommonProtocol as Quel1Box,
    )


class Quel1ConfigurationManager:
    """Handle configuration, define, and dump operations for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def dump_box(
        self,
        *,
        box_name: str,
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> dict:
        """
        Dump one box configuration and tolerate box-level errors.

        Parameters
        ----------
        box_name : str
            Target box name.
        check_box_availability : callable
            Validator for box availability.
        create_box : callable
            Fallback box factory called as `(box_name, reconnect)`.

        Returns
        -------
        dict
            Dumped box configuration.
        """
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
                check_box_availability=check_box_availability,
                create_box=create_box,
            )
            box_config = box.dump_box()
        except Exception:
            logger.exception(f"Failed to dump box {box_name}.")
            box_config = {}
        return box_config

    def dump_port(
        self,
        *,
        box_name: str,
        port_number: int | tuple[int, int],
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> dict:
        """
        Dump one port configuration and tolerate box-level errors.

        Parameters
        ----------
        box_name : str
            Target box name.
        port_number : int | tuple[int, int]
            Port number.
        check_box_availability : callable
            Validator for box availability.
        create_box : callable
            Fallback box factory called as `(box_name, reconnect)`.

        Returns
        -------
        dict
            Dumped port configuration.
        """
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
                check_box_availability=check_box_availability,
                create_box=create_box,
            )
            port_config = box.dump_port(port_number)
        except Exception:
            logger.exception(f"Failed to dump port {port_number} of box {box_name}.")
            port_config = {}
        return port_config

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
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> None:
        """
        Configure one box port.

        Parameters
        ----------
        box_name : str
            Target box name.
        port : int | tuple[int, int]
            Port number.
        lo_freq : float | None
            Local oscillator frequency in GHz.
        cnco_freq : float | None
            CNCO frequency in GHz.
        vatt : int | None
            VATT value.
        sideband : str | None
            Sideband value.
        fullscale_current : int | None
            Fullscale current value.
        rfswitch : str | None
            RF switch value.
        check_box_availability : callable
            Validator for box availability.
        create_box : callable
            Fallback box factory called as `(box_name, reconnect)`.
        """
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
            check_box_availability=check_box_availability,
            create_box=create_box,
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
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> None:
        """
        Configure one box channel.

        Parameters
        ----------
        box_name : str
            Target box name.
        port : int | tuple[int, int]
            Port number.
        channel : int
            Channel number.
        fnco_freq : float | None
            FNCO frequency in GHz.
        check_box_availability : callable
            Validator for box availability.
        create_box : callable
            Fallback box factory called as `(box_name, reconnect)`.
        """
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
            check_box_availability=check_box_availability,
            create_box=create_box,
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
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> None:
        """
        Configure one box runit.

        Parameters
        ----------
        box_name : str
            Target box name.
        port : int | tuple[int, int]
            Port number.
        runit : int
            Runit number.
        fnco_freq : float | None
            FNCO frequency in GHz.
        check_box_availability : callable
            Validator for box availability.
        create_box : callable
            Fallback box factory called as `(box_name, reconnect)`.
        """
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
            check_box_availability=check_box_availability,
            create_box=create_box,
        )
        box.config_runit(
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
        )

    @staticmethod
    def define_clockmaster(
        *,
        qubecalib: QubeCalib,
        ipaddr: str,
        reset: bool = True,
    ) -> None:
        """Define clockmaster in qubecalib."""
        qubecalib.define_clockmaster(ipaddr=ipaddr, reset=reset)

    @staticmethod
    def define_box(
        *,
        qubecalib: QubeCalib,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """Define one box in qubecalib."""
        qubecalib.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
        )

    @staticmethod
    def define_port(
        *,
        qubecalib: QubeCalib,
        port_name: str,
        box_name: str,
        port_number: int,
    ) -> None:
        """Define one port in qubecalib."""
        qubecalib.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
        )

    @staticmethod
    def define_channel(
        *,
        qubecalib: QubeCalib,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """Define one channel in qubecalib."""
        qubecalib.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    @staticmethod
    def add_channel_target_relation(
        *,
        qubecalib: QubeCalib,
        channel_name: str,
        target_name: str,
    ) -> None:
        """Add one channel-target relation if missing."""
        relation = (channel_name, target_name)
        sysdb = qubecalib.sysdb
        if relation not in sysdb._relation_channel_target:
            sysdb._relation_channel_target.append(relation)

    @staticmethod
    def define_target(
        *,
        qubecalib: QubeCalib,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ) -> None:
        """Define one target in qubecalib."""
        qubecalib.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    @staticmethod
    def modify_target_frequency(
        *,
        qubecalib: QubeCalib,
        target: str,
        frequency: float,
    ) -> None:
        """Modify one target frequency in qubecalib."""
        qubecalib.modify_target_frequency(target, frequency)

    def modify_target_frequencies(
        self,
        *,
        qubecalib: QubeCalib,
        frequencies: dict[str, float],
    ) -> None:
        """Modify multiple target frequencies in qubecalib."""
        for target, frequency in frequencies.items():
            self.modify_target_frequency(
                qubecalib=qubecalib,
                target=target,
                frequency=frequency,
            )

    @staticmethod
    def add_sequencer(*, qubecalib: QubeCalib, sequencer: Any) -> None:
        """Add one sequencer into qubecalib executor queue."""
        qubecalib._executor.add_command(sequencer)

    @staticmethod
    def show_command_queue(*, qubecalib: QubeCalib) -> str:
        """Return current qubecalib command queue string."""
        return qubecalib.show_command_queue()

    @staticmethod
    def clear_command_queue(*, qubecalib: QubeCalib) -> None:
        """Clear qubecalib command queue."""
        qubecalib.clear_command_queue()

    def _resolve_box(
        self,
        *,
        box_name: str,
        reconnect: bool,
        check_box_availability: Callable[[str], None],
        create_box: Callable[[str, bool], Quel1Box],
    ) -> Quel1Box:
        """Resolve a box from runtime context or create it lazily."""
        check_box_availability(box_name)
        boxpool = self._runtime_context.boxpool
        if boxpool is not None and box_name in boxpool._boxes:
            return boxpool._boxes[box_name][0]
        return create_box(box_name, reconnect)
