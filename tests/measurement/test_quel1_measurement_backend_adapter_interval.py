"""Tests for QuEL-1 adapter interval guard behavior."""

from __future__ import annotations

from types import SimpleNamespace
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
        return {target: [{}] for target in targets}  # shape is irrelevant here


class _ExperimentSystemStub:
    def __init__(self, sideband_by_target: dict[str, str]) -> None:
        self.sideband_by_target = sideband_by_target

    def get_target(self, target: str) -> SimpleNamespace:
        return SimpleNamespace(sideband=self.sideband_by_target[target])


def _make_config(interval: float) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=1,
        shot_interval=interval,
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
    assert hasattr(payload, "interval_ns")
    # block duration for strict QuEL-1 profile is 128 ns; one extra block is added.
    assert payload.interval_ns == 256


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
    assert hasattr(payload, "interval_ns")
    # ceil((128 + 256) / 128) * 128 = 384
    assert payload.interval_ns == 384


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


def test_build_execution_request_keeps_classification_off_without_options(
    monkeypatch,
) -> None:
    """Given a normal measurement config, backend payload should not enable DSP classification."""
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
        return {}, {"RQ00": object()}

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        _sampled_sequences.__get__(adapter, Quel1MeasurementBackendAdapter),
        raising=False,
    )

    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=_make_config(interval=0.0),
    )

    payload = request.payload
    assert payload.enable_classification is False
    assert payload.classification_lines is None


def test_build_execution_request_scales_gmm_linear_line_constants(monkeypatch) -> None:
    """Lower-sideband GMM lines should use backend coordinates and e7 units."""
    backend = _BackendControllerStub()
    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend),
            experiment_system=cast(
                Any,
                _ExperimentSystemStub(sideband_by_target={"RQ00": "L"}),
            ),
        ),
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _ = (self, schedule)
        return {}, {"RQ00": object()}

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        _sampled_sequences.__get__(adapter, Quel1MeasurementBackendAdapter),
        raising=False,
    )

    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = MeasurementConfig(
        n_shots=1,
        shot_interval=0.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        classification_source="gmm_linear",
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=config,
        quel1_options=Quel1MeasurementOptions(
            classification_line_param0={"RQ00": (1.0, 2.0, -0.25)},
            classification_line_param1={"RQ00": (1.0, -3.0, -0.5)},
        ),
    )

    payload = request.payload
    assert payload.classification_lines["RQ00"].line0 == (
        1.0,
        -2.0,
        -(1 << 16),
    )
    assert payload.classification_lines["RQ00"].line1 == (
        1.0,
        3.0,
        -(1 << 17),
    )


def test_build_execution_request_scales_gmm_linear_line_constants_without_demodulation(
    monkeypatch,
) -> None:
    """Given DSP demodulation disabled, GMM line constants use the full raw-I/Q scale."""
    backend = _BackendControllerStub()
    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend),
            experiment_system=cast(
                Any,
                _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
            ),
        ),
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _ = (self, schedule)
        return {}, {"RQ00": object()}

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        _sampled_sequences.__get__(adapter, Quel1MeasurementBackendAdapter),
        raising=False,
    )

    with PulseSchedule(["RQ00"]) as pulse_schedule:
        pulse_schedule.add("RQ00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )
    config = MeasurementConfig(
        n_shots=1,
        shot_interval=0.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        classification_source="gmm_linear",
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=config,
        quel1_options=Quel1MeasurementOptions(
            demodulation=False,
            classification_line_param0={"RQ00": (1.0, 2.0, -0.25)},
            classification_line_param1={"RQ00": (1.0, -3.0, -0.5)},
        ),
    )

    payload = request.payload
    assert payload.classification_lines["RQ00"].line0 == (
        1.0,
        2.0,
        -(1 << 30),
    )
    assert payload.classification_lines["RQ00"].line1 == (
        1.0,
        -3.0,
        -(1 << 31),
    )
