"""Tests for Quel1Box compatibility adapter across quelware API generations."""

from __future__ import annotations

from qubex.backend.quel1.quel1_box_compat import adapt_quel1_box


class _LegacyBox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def initialize_all_awgs(self) -> None:
        self.calls.append(("initialize_all_awgs", None))

    def initialize_all_capunits(self) -> None:
        self.calls.append(("initialize_all_capunits", None))

    def read_current_and_latched_clock(self) -> tuple[int, int]:
        return (123, 456)

    def start_emission(self, channels: set[tuple[int, int]]) -> None:
        self.calls.append(("start_emission", channels))

    def reserve_emission(self, channels: set[tuple[int, int]], timecounter: int) -> None:
        self.calls.append(("reserve_emission", (channels, timecounter)))


class _ModernBox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def initialize_all_awgunits(self) -> None:
        self.calls.append(("initialize_all_awgunits", None))

    def initialize_all_capunits(self) -> None:
        self.calls.append(("initialize_all_capunits", None))

    def get_current_timecounter(self) -> int:
        return 1000

    def get_latest_sysref_timecounter(self) -> int:
        return 2000

    def start_wavegen(
        self,
        channels: set[tuple[int, int]],
        timecounter: int | None = None,
    ) -> object:
        self.calls.append(("start_wavegen", (channels, timecounter)))

        class _Task:
            def result(self, timeout: float | None = None) -> None:
                _ = timeout
                return None

        return _Task()


def test_adapter_maps_legacy_methods() -> None:
    """Given legacy box, adapter maps unified methods to legacy operations."""
    legacy = _LegacyBox()
    box = adapt_quel1_box(legacy)

    box.initialize_all_awgunits()
    box.initialize_all_capunits()
    assert box.get_current_timecounter() == 123
    assert box.get_latest_sysref_timecounter() == 456

    task_now = box.start_wavegen({(1, 0)})
    task_now.result()
    task_timed = box.start_wavegen({(1, 0)}, timecounter=999)
    task_timed.result()

    assert ("initialize_all_awgs", None) in legacy.calls
    assert ("initialize_all_capunits", None) in legacy.calls
    assert ("start_emission", {(1, 0)}) in legacy.calls
    assert ("reserve_emission", ({(1, 0)}, 999)) in legacy.calls


def test_adapter_passthroughs_modern_methods() -> None:
    """Given modern box, adapter delegates to modern API methods."""
    modern = _ModernBox()
    box = adapt_quel1_box(modern)

    box.initialize_all_awgunits()
    box.initialize_all_capunits()
    assert box.get_current_timecounter() == 1000
    assert box.get_latest_sysref_timecounter() == 2000

    task = box.start_wavegen({(2, 0)}, timecounter=10_000)
    task.result()
    assert ("start_wavegen", ({(2, 0)}, 10_000)) in modern.calls

