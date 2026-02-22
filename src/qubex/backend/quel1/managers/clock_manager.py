"""Clock synchronization manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class Quel1ClockManager:
    """Handle read/check/resync operations for backend clocks."""

    def read_clocks(
        self,
        *,
        box_list: list[str],
        check_box_availability: Callable[[str], None],
        resolve_box_sss_ip: Callable[[str], str],
        read_clock_at_ip: Callable[[str], tuple[bool, int, int]],
    ) -> list[tuple[bool, int, int]]:
        """
        Read sequencer clocks for selected boxes.

        Parameters
        ----------
        box_list : list[str]
            Box names.
        check_box_availability : Callable[[str], None]
            Box availability validator.
        resolve_box_sss_ip : Callable[[str], str]
            Resolver from box name to sequencer IP address.
        read_clock_at_ip : Callable[[str], tuple[bool, int, int]]
            Sequencer-clock reader at one IP address.

        Returns
        -------
        list[tuple[bool, int, int]]
            Sequencer clock readings.
        """
        result: list[tuple[bool, int, int]] = []
        for box_name in box_list:
            check_box_availability(box_name)
            result.append(read_clock_at_ip(resolve_box_sss_ip(box_name)))
        return result

    def check_clocks(
        self, *, read_clocks: Callable[[], list[tuple[bool, int, int]]]
    ) -> bool:
        """
        Check whether clocks are synchronized.

        Parameters
        ----------
        read_clocks : Callable[[], list[tuple[bool, int, int]]]
            Clock-reading function for target boxes.

        Returns
        -------
        bool
            True when clocks are synchronized.
        """
        result = read_clocks()
        timestamps: list[str] = []
        accuracy = -8
        for _, clock, sysref_latch in result:
            timestamps.append(str(clock)[:accuracy])
            timestamps.append(str(sysref_latch)[:accuracy])
        return len(set(timestamps)) == 1

    def resync_clocks(
        self,
        *,
        box_list: list[str],
        get_connected_clockmaster: Callable[[], Any | None],
        get_clockmaster_setting_ipaddr: Callable[[], str | None],
        create_clockmaster_client: Callable[[str], Any],
        resolve_box_sss_ip: Callable[[str], str],
        check_clocks: Callable[[list[str]], bool],
    ) -> bool:
        """
        Resynchronize clocks for selected boxes.

        Parameters
        ----------
        box_list : list[str]
            Box names.
        get_connected_clockmaster : Callable[[], Any | None]
            Connected clockmaster getter.
        get_clockmaster_setting_ipaddr : Callable[[], str | None]
            Configured clockmaster IP getter.
        create_clockmaster_client : Callable[[str], Any]
            Clockmaster client factory.
        resolve_box_sss_ip : Callable[[str], str]
            Resolver from box name to sequencer IP address.
        check_clocks : Callable[[list[str]], bool]
            Clock synchronization checker.

        Returns
        -------
        bool
            True when synchronization succeeded.
        """
        if len(box_list) < 2:
            return True
        master = get_connected_clockmaster()
        if master is None:
            clockmaster_ipaddr = get_clockmaster_setting_ipaddr()
            if clockmaster_ipaddr is None:
                raise ValueError("clock master is not found")
            master = create_clockmaster_client(clockmaster_ipaddr)
        master.kick_clock_synch([resolve_box_sss_ip(box_name) for box_name in box_list])
        return check_clocks(box_list)

    def reset_clockmaster(
        self,
        *,
        ipaddr: str,
        get_connected_clockmaster: Callable[[], Any | None],
        get_clockmaster_setting_ipaddr: Callable[[], str | None],
        create_clockmaster_client: Callable[[str], Any],
    ) -> bool:
        """
        Reset the clockmaster endpoint.

        Parameters
        ----------
        ipaddr : str
            Clockmaster IP address.
        get_connected_clockmaster : Callable[[], Any | None]
            Connected clockmaster getter.
        get_clockmaster_setting_ipaddr : Callable[[], str | None]
            Configured clockmaster IP getter.
        create_clockmaster_client : Callable[[str], Any]
            Clockmaster client factory.

        Returns
        -------
        bool
            True when reset succeeded.
        """
        connected_master = get_connected_clockmaster()
        configured_ipaddr = get_clockmaster_setting_ipaddr()
        if connected_master is not None and configured_ipaddr == ipaddr:
            return connected_master.reset()
        return create_clockmaster_client(ipaddr).reset()

    def sync_clocks(self, *, resync_clocks: Callable[[], bool], box_count: int) -> bool:
        """
        Ensure clocks are synchronized, performing resync when needed.

        Parameters
        ----------
        resync_clocks : Callable[[], bool]
            Resynchronization action.
        box_count : int
            Number of boxes in target scope.

        Returns
        -------
        bool
            True when synchronized.
        """
        if box_count < 2:
            return True
        synchronized = resync_clocks()
        if not synchronized:
            logger.warning("Failed to synchronize clocks.")
        return synchronized
