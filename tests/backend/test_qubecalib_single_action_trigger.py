"""Regression tests for driver single.Action triggered capture flow."""

from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode

try:
    _e7compat = importlib.import_module("qxdriver_quel.e7compat")
    _single = importlib.import_module("qxdriver_quel.instrument.quel.quel1.driver.single")
except ModuleNotFoundError:
    _e7compat = importlib.import_module("qubecalib.e7compat")
    _single = importlib.import_module("qubecalib.instrument.quel.quel1.driver.single")

CaptureParam = _e7compat.CaptureParam
WaveSequence = _e7compat.WaveSequence
Action = _single.Action
AwgId = _single.AwgId
RunitId = _single.RunitId


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


class _FakeBox:
    def __init__(self) -> None:
        self.start_capture_by_awg_trigger_calls: list[tuple[set, set]] = []
        self.start_wavegen_calls: list[set] = []

    def start_capture_by_awg_trigger(
        self,
        *,
        runits: set[tuple[int, int]],
        channels: set[tuple[int, int]],
    ) -> tuple[_Task, _Task]:
        self.start_capture_by_awg_trigger_calls.append((runits, channels))
        readers = {(1, 0): _Reader(1 + 0j)}
        return _Task(readers), _Task(None)

    def start_wavegen(self, channels: set[tuple[int, int]]) -> _Task:
        self.start_wavegen_calls.append(channels)
        return _Task(None)


def test_triggered_capture_does_not_start_wavegen_twice() -> None:
    """Triggered capture action must not call start_wavegen after trigger start."""
    box = _FakeBox()
    action = Action(
        box=cast(Any, box),
        wseqs=MappingProxyType({AwgId(port=0, channel=0): WaveSequence(0, 1)}),
        cprms=MappingProxyType({RunitId(port=1, runit=0): CaptureParam()}),
        triggers=MappingProxyType({1: AwgId(port=0, channel=0)}),
    )

    status, data = action.action()

    assert len(box.start_capture_by_awg_trigger_calls) == 1
    assert box.start_wavegen_calls == []
    assert status == {1: CaptureReturnCode.SUCCESS}
    assert (1, 0) in data
