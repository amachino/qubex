"""Tests for QuEL-1 adapter interval guard behavior."""

from __future__ import annotations

from typing import Any, cast

from qxpulse import Blank, PulseSchedule

from qubex.measurement.adapters.backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions


class _BackendControllerStub:
    def __init__(self) -> None:
        self.targets: list[str] = []

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict[str, str]]]:
        self.targets = targets
        return {"Q00": [{}]}  # shape is irrelevant for this unit test


def _make_config(interval: float) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=1,
        shot_interval_ns=interval,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
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


def test_build_execution_request_honors_quel1_dsp_demodulation_option(
    monkeypatch,
) -> None:
    """Given explicit QuEL-1 options, when building request, then DSP demodulation flag follows the option."""
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
        _ = (self, schedule)
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
        config=_make_config(interval=0.0),
        quel1_options=Quel1MeasurementOptions(demodulation=False),
    )

    payload = request.payload
    assert hasattr(payload, "dsp_demodulation")
    assert payload.dsp_demodulation is False
