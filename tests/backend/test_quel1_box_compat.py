"""Tests for Quel1Box compatibility adapter across quelware API generations."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from qubex.backend.quel1.quel1_box_adapter import adapt_quel1_box


class _Task:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def result(self, timeout: float | None = None) -> object:
        _ = timeout
        return self._payload

    def cancel(self) -> bool:
        return False

    def done(self) -> bool:
        return True


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

    def reserve_emission(
        self, channels: set[tuple[int, int]], timecounter: int
    ) -> None:
        self.calls.append(("reserve_emission", (channels, timecounter)))

    def capture_start(
        self,
        *,
        port: int,
        runits: list[int],
        triggering_channel: tuple[int, int] | None = None,
    ) -> _Task:
        self.calls.append(("capture_start", (port, tuple(runits), triggering_channel)))
        payload = (
            "SUCCESS",
            {r: np.asarray([complex(port, r)], dtype=np.complex64) for r in runits},
        )
        return _Task(payload)


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

    def start_capture_now(self, runits: set[tuple[int, int]]) -> _Task:
        self.calls.append(("start_capture_now", runits))
        return _Task({"modern": "capture_now"})

    def start_capture_by_awg_trigger(
        self,
        *,
        runits: set[tuple[int, int]],
        channels: set[tuple[int, int]],
        timecounter: int | None = None,
    ) -> tuple[_Task, _Task]:
        self.calls.append(
            ("start_capture_by_awg_trigger", (runits, channels, timecounter))
        )
        return _Task({"modern": "cap"}), _Task({"modern": "gen"})


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


def test_adapter_maps_legacy_capture_methods() -> None:
    """Given legacy box, adapter emulates capture_now and capture_by_awg_trigger."""
    legacy = _LegacyBox()
    box = adapt_quel1_box(legacy)

    task = box.start_capture_now({(1, 0), (1, 2), (2, 1)})
    readers = cast(dict[tuple[int, int], Any], task.result())
    assert sorted(readers) == [(1, 0), (1, 2), (2, 1)]
    assert np.allclose(readers[(1, 0)].rawwave(), np.asarray([1 + 0j], np.complex64))
    assert np.allclose(readers[(2, 1)].rawwave(), np.asarray([2 + 1j], np.complex64))

    cap_task, gen_task = box.start_capture_by_awg_trigger(
        runits={(1, 0), (2, 1)},
        channels={(10, 3), (11, 0)},
        timecounter=1234,
    )
    triggered = cast(dict[tuple[int, int], Any], cap_task.result())
    gen_task.result()
    assert sorted(triggered) == [(1, 0), (2, 1)]

    capture_calls = cast(
        list[tuple[str, tuple[int, tuple[int, ...], tuple[int, int] | None]]],
        [c for c in legacy.calls if c[0] == "capture_start"],
    )
    assert len(capture_calls) == 4
    trigger_params = capture_calls[-2:]
    assert trigger_params[0][1][2] in {(10, 3), (11, 0)}
    assert trigger_params[1][1][2] in {(10, 3), (11, 0)}
    assert ("reserve_emission", ({(10, 3), (11, 0)}, 1234)) in legacy.calls


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


def test_adapter_passthroughs_modern_capture_methods() -> None:
    """Given modern box, adapter delegates capture APIs without conversion."""
    modern = _ModernBox()
    box = adapt_quel1_box(modern)

    cap_now = box.start_capture_now({(1, 0)})
    assert cap_now.result() == {"modern": "capture_now"}

    cap_task, gen_task = box.start_capture_by_awg_trigger(
        runits={(1, 0)},
        channels={(5, 2)},
        timecounter=77,
    )
    assert cap_task.result() == {"modern": "cap"}
    assert gen_task.result() == {"modern": "gen"}
    assert ("start_capture_now", {(1, 0)}) in modern.calls
    assert (
        "start_capture_by_awg_trigger",
        ({(1, 0)}, {(5, 2)}, 77),
    ) in modern.calls
