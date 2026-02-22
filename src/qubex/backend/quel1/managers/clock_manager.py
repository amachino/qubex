# ruff: noqa: SLF001

"""Clock synchronization manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import QuBEMasterClientProtocol


class Quel1ClockManager:
    """Handle read/check/resync operations for backend clocks."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def read_clocks(self, *, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """Read sequencer clocks for selected boxes."""
        result: list[tuple[bool, int, int]] = []
        for box_name in box_list:
            self._runtime_context.validate_box_availability(box_name)
            result.append(self._read_clock_at_ip(self._resolve_box_sss_ip(box_name)))
        return result

    def check_clocks(self, *, box_list: list[str]) -> bool:
        """Check whether clocks are synchronized."""
        result = self.read_clocks(box_list=box_list)
        timestamps: list[str] = []
        accuracy = -8
        for _, clock, sysref_latch in result:
            timestamps.append(str(clock)[:accuracy])
            timestamps.append(str(sysref_latch)[:accuracy])
        return len(set(timestamps)) == 1

    def resync_clocks(self, *, box_list: list[str]) -> bool:
        """Resynchronize clocks for selected boxes."""
        if len(box_list) < 2:
            return True
        master = self._get_connected_clockmaster()
        if master is None:
            clockmaster_ipaddr = self._get_clockmaster_setting_ipaddr()
            if clockmaster_ipaddr is None:
                raise ValueError("clock master is not found")
            master = self._runtime_context.driver.QuBEMasterClient(clockmaster_ipaddr)
        master.kick_clock_synch(
            [self._resolve_box_sss_ip(box_name) for box_name in box_list]
        )
        return self.check_clocks(box_list=box_list)

    def reset_clockmaster(self, *, ipaddr: str) -> bool:
        """Reset the clockmaster endpoint."""
        connected_master = self._get_connected_clockmaster()
        configured_ipaddr = self._get_clockmaster_setting_ipaddr()
        if connected_master is not None and configured_ipaddr == ipaddr:
            return connected_master.reset()
        return self._runtime_context.driver.QuBEMasterClient(ipaddr).reset()

    def sync_clocks(self, *, box_list: list[str]) -> bool:
        """Ensure clocks are synchronized, performing resync when needed."""
        if len(box_list) < 2:
            return True
        synchronized = self.resync_clocks(box_list=box_list)
        if not synchronized:
            logger.warning("Failed to synchronize clocks.")
        return synchronized

    def _get_connected_clockmaster(self) -> QuBEMasterClientProtocol | None:
        """Return clockmaster from connected runtime system when available."""
        quel1system = self._runtime_context.quel1system_or_none()
        if quel1system is None:
            return None
        return quel1system._clockmaster

    def _resolve_box_sss_ip(self, box_name: str) -> str:
        """Resolve sequencer-service IP address for one box."""
        db = self._runtime_context.qubecalib.system_config_database
        return str(db._box_settings[box_name].ipaddr_sss)

    def _read_clock_at_ip(self, ipaddr_sss: str) -> tuple[bool, int, int]:
        """Read one sequencer clock tuple by IP address."""
        return self._runtime_context.driver.SequencerClient(
            target_ipaddr=ipaddr_sss
        ).read_clock()

    def _get_clockmaster_setting_ipaddr(self) -> str | None:
        """Return configured clockmaster IP address if available."""
        db = self._runtime_context.qubecalib.system_config_database
        clockmaster_setting = db._clockmaster_setting
        if clockmaster_setting is None:
            return None
        return str(clockmaster_setting.ipaddr)
