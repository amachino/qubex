"""Tests for QuEL-1 adapter interval guard behavior."""

from __future__ import annotations

from typing import Any, cast

from qxpulse import Blank, PulseSchedule

from qubex.measurement.adapters.backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule


class _BackendControllerStub:
    def __init__(self) -> None:
        self.targets: list[str] = []

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict[str, str]]]:
        self.targets = targets
        return {"Q00": [{}]}  # shape is irrelevant for this unit test


def _make_config(interval: float) -> MeasurementConfig:
    return MeasurementConfig(
        mode="avg",
        shots=1,
        interval=interval,
        frequencies={},
        enable_dsp_demodulation=True,
        enable_dsp_sum=False,
        enable_dsp_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_build_execution_request_adds_one_block_margin_when_interval_nonpositive(
    monkeypatch,
) -> None:
    """Given non-positive interval, when building request, then one block margin is added."""
    backend = _BackendControllerStub()
    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend),
            experiment_system=cast(Any, object()),
        ),
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _ = schedule
        return {"Q00": object()}, {}

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        _sampled_sequences.__get__(adapter, Quel1MeasurementBackendAdapter),
        raising=False,
    )

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    request = adapter.build_execution_request(
        schedule=schedule, config=_make_config(interval=0.0)
    )

    payload = request.payload
    assert hasattr(payload, "interval")
    # block duration for strict QuEL-1 profile is 128 ns; one extra block is added.
    assert payload.interval == 256


def test_build_execution_request_preserves_positive_interval_without_extra_margin(
    monkeypatch,
) -> None:
    """Given positive interval, when building request, then interval follows duration-plus-interval alignment."""
    backend = _BackendControllerStub()
    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend),
            experiment_system=cast(Any, object()),
        ),
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _ = schedule
        return {"Q00": object()}, {}

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        _sampled_sequences.__get__(adapter, Quel1MeasurementBackendAdapter),
        raising=False,
    )

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=_make_config(interval=256.0),
    )

    payload = request.payload
    assert hasattr(payload, "interval")
    # ceil((128 + 256) / 128) * 128 = 384
    assert payload.interval == 384
