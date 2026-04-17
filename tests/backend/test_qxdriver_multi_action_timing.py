"""Regression tests for qxdriver triggered multi-box timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode
from qxdriver_quel1.driver import multi
from qxdriver_quel1.driver.single import (
    _TRIGGERED_CAPTURE_KEY,
    Action as SingleAction,
    AwgId,
    RunitId,
)
from qxdriver_quel1.e7awg.compat import CaptureParam, WaveSequence


class _Reader:
    def __init__(self, value: complex) -> None:
        self._value = value

    def rawwave(self) -> np.ndarray:
        return np.asarray([self._value], dtype=np.complex64)


class _Task:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def result(self, timeout: float | None = None) -> Any:
        _ = timeout
        return self._payload


class _FailingTask:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def result(self, timeout: float | None = None) -> Any:
        _ = timeout
        raise self._error


class _CancelableCaptureTask:
    def __init__(self) -> None:
        self.cancel_calls: int = 0

    def result(self, timeout: float | None = None) -> Any:
        _ = timeout
        raise AssertionError(
            "capture task should not be awaited after trigger AWG too-late failure"
        )

    def cancel(
        self,
        timeout: float | None = None,
        polling_period: float | None = None,
    ) -> bool:
        _ = timeout, polling_period
        self.cancel_calls += 1
        return True


class _TriggeredCaptureBox:
    def __init__(self) -> None:
        self.calls: list[
            tuple[set[tuple[int, int]], set[tuple[int, int]], int | None]
        ] = []

    def start_capture_by_awg_trigger(
        self,
        *,
        runits: set[tuple[int, int]],
        channels: set[tuple[int, int]],
        timecounter: int | None = None,
    ) -> tuple[_Task, _Task]:
        self.calls.append((runits, channels, timecounter))
        readers = {(1, 0): _Reader(1 + 0j)}
        return _Task(readers), _Task(None)


def test_single_action_capture_start_forwards_scheduled_timecounter() -> None:
    """Triggered single action should forward scheduled time to capture start."""
    box = _TriggeredCaptureBox()
    action = SingleAction(
        box=cast(Any, box),
        wseqs=MappingProxyType({AwgId(port=0, channel=0): WaveSequence(0, 1)}),
        cprms=MappingProxyType({RunitId(port=1, runit=0): CaptureParam()}),
        triggers=MappingProxyType({1: AwgId(port=0, channel=0)}),
    )

    futures = action.capture_start(timecounter=1234)
    status, data = action.capture_stop(futures)

    assert _TRIGGERED_CAPTURE_KEY in futures
    assert box.calls == [({(1, 0)}, {(0, 0)}, 1234)]
    assert status == {1: CaptureReturnCode.SUCCESS}
    assert (1, 0) in data


def test_single_action_capture_stop_surfaces_trigger_awg_failure_first() -> None:
    """Triggered single action should surface AWG scheduling failures before capture waits."""
    action = SingleAction(
        box=cast(Any, object()),
        wseqs=MappingProxyType({AwgId(port=0, channel=0): WaveSequence(0, 1)}),
        cprms=MappingProxyType({RunitId(port=1, runit=0): CaptureParam()}),
        triggers=MappingProxyType({1: AwgId(port=0, channel=0)}),
    )

    futures = {
        _TRIGGERED_CAPTURE_KEY: (
            _FailingTask(AssertionError("capture task should not be awaited first")),
            _FailingTask(
                RuntimeError("specified timecount (= 1) is too late to schedule")
            ),
        )
    }

    with pytest.raises(RuntimeError, match="too late to schedule"):
        action.capture_stop(cast(Any, futures))


def test_single_action_capture_stop_cancels_capture_task_on_trigger_awg_too_late() -> (
    None
):
    """Triggered single action should best-effort cancel the paired capture task on too-late failure."""
    cap_task = _CancelableCaptureTask()
    action = SingleAction(
        box=cast(Any, object()),
        wseqs=MappingProxyType({AwgId(port=0, channel=0): WaveSequence(0, 1)}),
        cprms=MappingProxyType({RunitId(port=1, runit=0): CaptureParam()}),
        triggers=MappingProxyType({1: AwgId(port=0, channel=0)}),
    )

    futures = {
        _TRIGGERED_CAPTURE_KEY: (
            cap_task,
            _FailingTask(
                RuntimeError("specified timecount (= 1) is too late to schedule")
            ),
        )
    }

    with pytest.raises(RuntimeError, match="too late to schedule"):
        action.capture_stop(cast(Any, futures))

    assert cap_task.cancel_calls == 1


@dataclass
class _CompletedWavegenTask:
    def result(self, timeout: float | None = None) -> None:
        _ = timeout


@dataclass
class _FakeWavegenBox:
    current_time: int = 100
    latest_sysref_time: int = 0
    reservations: list[tuple[set[tuple[str, int]], int | None]] = field(
        default_factory=list
    )

    def get_current_timecounter(self) -> int:
        return self.current_time

    def get_latest_sysref_timecounter(self) -> int:
        return self.latest_sysref_time

    def start_wavegen(
        self,
        channels: set[tuple[str, int]],
        timecounter: int | None = None,
    ) -> _CompletedWavegenTask:
        self.reservations.append((channels, timecounter))
        return _CompletedWavegenTask()


@dataclass
class _FakeTriggeredAction:
    box: _FakeWavegenBox
    capture_params: dict[str, object]
    wave_sequences: list[SimpleNamespace]
    trigger_settings: dict[str, object]
    capture_start_timecounter: int | None = None

    def capture_start(self, *, timecounter: int | None = None) -> dict[str, str]:
        self.capture_start_timecounter = timecounter
        return {"P1": "future"}

    def capture_stop(
        self,
        futures: dict[str, str],
    ) -> tuple[
        dict[str, CaptureReturnCode],
        dict[tuple[str, int], np.ndarray],
    ]:
        _ = futures
        return {"P1": CaptureReturnCode.SUCCESS}, {
            ("P1", 0): np.asarray([1 + 0j], dtype=np.complex64)
        }


@dataclass
class _FakeAwgOnlyAction:
    box: _FakeWavegenBox
    wave_sequences: list[SimpleNamespace]
    trigger_settings: dict[str, object] = field(default_factory=dict)
    capture_params: dict[str, object] = field(default_factory=dict)

    def capture_start(self, *, timecounter: int | None = None) -> dict[str, str]:
        _ = timecounter
        return {}

    def capture_stop(
        self,
        futures: dict[str, str],
    ) -> tuple[dict[str, CaptureReturnCode], dict[tuple[str, int], np.ndarray]]:
        _ = futures
        return {}, {}


def test_multi_action_arms_triggered_boxes_with_scheduled_time() -> None:
    """Triggered multi action should arm triggered boxes at the shared schedule."""
    monitor_box = _FakeWavegenBox()
    target_box = _FakeWavegenBox()
    monitor_action = _FakeTriggeredAction(
        box=monitor_box,
        capture_params={"P1": object()},
        wave_sequences=[SimpleNamespace(port="TP", channel=0)],
        trigger_settings={"P1": object()},
    )
    target_action = _FakeAwgOnlyAction(
        box=target_box,
        wave_sequences=[SimpleNamespace(port="OUT", channel=1)],
    )
    action = multi.Action(
        quel1system=cast(
            Any,
            SimpleNamespace(
                box={"MON": monitor_box, "GEN": target_box},
                timing_shift={"MON": 0, "GEN": 0},
                displacement=0,
            ),
        ),
        actions=cast(
            Any,
            MappingProxyType({"MON": monitor_action, "GEN": target_action}),
        ),
        estimated_timediff=MappingProxyType({"MON": 0, "GEN": 0}),
        reference_box_name="MON",
        ref_sysref_time_offset=0,
    )

    status, data = action.action()

    assert monitor_action.capture_start_timecounter is not None
    assert target_box.reservations == [
        ({("OUT", 1)}, monitor_action.capture_start_timecounter)
    ]
    assert monitor_box.reservations == []
    assert status == {("MON", "P1"): CaptureReturnCode.SUCCESS}
    assert ("MON", "P1", 0) in data
