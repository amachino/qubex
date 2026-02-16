"""Tests for qxdriver_quel clockmaster compatibility wrappers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from qxdriver_quel.clockmaster import compat


@dataclass(eq=False)
class _FakeBox:
    _sss_ipaddr: str

    def __post_init__(self) -> None:
        self.wss = SimpleNamespace(ipaddr_sss=self._sss_ipaddr)


class _FakeQuelClockMasterV1:
    instances: ClassVar[list[_FakeQuelClockMasterV1]] = []

    def __init__(
        self,
        ipaddr: str,
        boxes: Collection[Any],
        allow_automatic_boot: bool = True,
    ) -> None:
        self.ipaddr = ipaddr
        self.allow_automatic_boot = allow_automatic_boot
        self._boxes = set(boxes)
        self.sync_calls = 0
        self.terminate_calls = 0
        self.counter = 123_456
        self.instances.append(self)

    def sync_boxes(self) -> None:
        """Record one sync request."""
        self.sync_calls += 1

    def get_current_timecounter(self) -> int:
        """Return a deterministic fake counter."""
        return self.counter

    def terminate(self) -> None:
        """Record one terminate request."""
        self.terminate_calls += 1

    @property
    def box_count(self) -> int:
        """Return the number of boxes currently associated with the master."""
        return len(self._boxes)


def test_qube_master_client_reuses_cached_clockmaster_for_sync_and_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one client, kick/read reuse one cached clock-master instance."""
    _FakeQuelClockMasterV1.instances = []
    monkeypatch.setattr(compat, "QuelClockMasterV1", _FakeQuelClockMasterV1)
    monkeypatch.setattr(compat, "_BOX_BY_SSS_IPADDR", {})
    monkeypatch.setattr(compat, "_SHARED_CLOCKMASTERS", {})
    compat.register_box(cast(Any, _FakeBox("10.0.1.1")))
    compat.register_box(cast(Any, _FakeBox("10.0.1.2")))

    client = compat.QuBEMasterClient(master_ipaddr="192.0.2.1")
    assert len(_FakeQuelClockMasterV1.instances) == 1
    clockmaster = _FakeQuelClockMasterV1.instances[0]

    client.kick_clock_synch(["10.0.1.1", "10.0.1.2"])
    first = client.read_clock()
    second = client.read_clock()

    assert len(_FakeQuelClockMasterV1.instances) == 1
    assert clockmaster.sync_calls == 1
    assert clockmaster.box_count == 2
    assert first == (True, 123_456)
    assert second == (True, 123_456)


def test_qube_master_client_kick_raises_for_unknown_box_without_recreating_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given unknown SSS IP, kick raises without recreating clock-master."""
    _FakeQuelClockMasterV1.instances = []
    monkeypatch.setattr(compat, "QuelClockMasterV1", _FakeQuelClockMasterV1)
    monkeypatch.setattr(compat, "_BOX_BY_SSS_IPADDR", {})
    monkeypatch.setattr(compat, "_SHARED_CLOCKMASTERS", {})

    client = compat.QuBEMasterClient(master_ipaddr="192.0.2.1")
    assert len(_FakeQuelClockMasterV1.instances) == 1
    clockmaster = _FakeQuelClockMasterV1.instances[0]

    with pytest.raises(RuntimeError, match="not registered"):
        client.kick_clock_synch(["10.0.1.99"])

    assert len(_FakeQuelClockMasterV1.instances) == 1
    assert clockmaster.sync_calls == 0


def test_qube_master_client_shares_clockmaster_between_clients_with_same_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given same IP, clients share one clock-master session."""
    _FakeQuelClockMasterV1.instances = []
    monkeypatch.setattr(compat, "QuelClockMasterV1", _FakeQuelClockMasterV1)
    monkeypatch.setattr(compat, "_SHARED_CLOCKMASTERS", {})

    first = compat.QuBEMasterClient(master_ipaddr="192.0.2.1")
    second = compat.QuBEMasterClient(master_ipaddr="192.0.2.1")

    assert len(_FakeQuelClockMasterV1.instances) == 1
    clockmaster = _FakeQuelClockMasterV1.instances[0]

    first.close()
    assert clockmaster.terminate_calls == 0

    second.close()
    assert clockmaster.terminate_calls == 1


def test_qube_master_client_uses_distinct_clockmasters_for_different_ips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given different IPs, clients create distinct clock-master sessions."""
    _FakeQuelClockMasterV1.instances = []
    monkeypatch.setattr(compat, "QuelClockMasterV1", _FakeQuelClockMasterV1)
    monkeypatch.setattr(compat, "_SHARED_CLOCKMASTERS", {})

    first = compat.QuBEMasterClient(master_ipaddr="192.0.2.1")
    second = compat.QuBEMasterClient(master_ipaddr="192.0.2.2")

    assert len(_FakeQuelClockMasterV1.instances) == 2

    first.close()
    second.close()
