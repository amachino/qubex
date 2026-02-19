"""Tests for Quel3 backend executor."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.backend import BackendExecutionRequest
from qubex.measurement.adapters import (
    Quel3BackendExecutor,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
    Quel3WaveformEvent,
)


def _make_payload() -> Quel3ExecutionPayload:
    timeline = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        events=(
            Quel3WaveformEvent(
                start_offset_ns=0.0,
                waveform=np.array([0.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            ),
        ),
        capture_windows=(),
        length_ns=0.4,
        modulation_frequency_hz=100_000_000.0,
    )
    return Quel3ExecutionPayload(
        timelines={"RQ00": timeline},
        instrument_aliases={"RQ00": "RQ00"},
        output_target_labels={"RQ00": "Q00"},
        interval_ns=10.0,
        repeats=16,
        mode="avg",
        dsp_demodulation=True,
        enable_sum=False,
        enable_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_quel3_backend_executor_delegates_to_backend_controller() -> None:
    """Given valid payload, when executing, then backend controller hook is called."""
    called: dict[str, object] = {}
    expected = object()

    class _Controller:
        def execute_measurement(self, *, payload: Quel3ExecutionPayload) -> object:
            called["payload"] = payload
            return expected

    payload = _make_payload()
    executor = Quel3BackendExecutor(backend_controller=_Controller())

    result = executor.execute(request=BackendExecutionRequest(payload=payload))

    assert called["payload"] is payload
    assert result is expected


def test_quel3_backend_executor_rejects_non_quel3_payload() -> None:
    """Given non-Quel3 payload, when executing, then TypeError is raised."""
    executor = Quel3BackendExecutor(backend_controller=object())

    with pytest.raises(TypeError, match="Quel3BackendExecutor expects"):
        executor.execute(request=BackendExecutionRequest(payload=object()))


def test_quel3_backend_executor_requires_backend_hook() -> None:
    """Given missing backend hook, when executing, then TypeError is raised."""
    payload = _make_payload()
    executor = Quel3BackendExecutor(backend_controller=object())

    with pytest.raises(TypeError, match="execute_measurement"):
        executor.execute(request=BackendExecutionRequest(payload=payload))
