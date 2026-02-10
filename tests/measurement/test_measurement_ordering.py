"""Tests for ordering behavior in measurement internals."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

from qxpulse import FlatTop, PulseSchedule

from qubex.backend import ControlParams, Target
from qubex.measurement.measurement_backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import (
    DspConfig,
    FrequencyConfig,
    MeasurementConfig,
)
from qubex.measurement.models.measurement_schedule import MeasurementSchedule


def test_schedule_builder_keeps_readout_target_order(monkeypatch) -> None:
    """Given schedule labels, when adding readout targets, then first-seen label order is kept."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParams,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1, "RQ01": 0.1}),
        ),
        pulse_factory=cast(
            MeasurementPulseFactory,
            SimpleNamespace(
                readout_pulse=lambda **_: FlatTop(duration=16, amplitude=0.1, tau=4)
            ),
        ),
        targets=cast(
            dict[str, Target],
            {
                "Q01": SimpleNamespace(is_pump=False, is_read=False),
                "Q00": SimpleNamespace(is_pump=False, is_read=False),
            },
        ),
        mux_dict={},
    )

    captured: dict[str, Any] = {}

    def _capture(
        self: MeasurementScheduleBuilder,
        *,
        schedule: PulseSchedule,
        readout_targets: list[str],
    ) -> CaptureSchedule:
        captured["readout_targets"] = readout_targets
        return CaptureSchedule(captures=[])

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(_capture, builder),
    )

    with PulseSchedule(["Q01", "Q00", "Q01"]) as schedule:
        pass

    builder.build(schedule=schedule, add_last_measurement=True)

    assert captured["readout_targets"] == ["RQ01", "RQ00"]


def test_backend_adapter_keeps_target_merge_order(monkeypatch) -> None:
    """Given gen and cap targets, when building request, then target merge order is deterministic."""

    class _BackendController:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def get_resource_map(self, targets: list[str]) -> dict[str, Any]:
            self.targets = targets
            return {}

        def create_quel1_sequencer(self, **_: Any) -> object:
            return object()

    adapter = cast(Any, object.__new__(Quel1MeasurementBackendAdapter))
    backend_controller = _BackendController()
    monkeypatch.setattr(
        adapter, "_backend_controller", backend_controller, raising=False
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return ({"Q01": object(), "Q00": object()}, {"Q00": object(), "Q02": object()})

    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        MethodType(_sampled_sequences, adapter),
        raising=False,
    )

    config = MeasurementConfig(
        mode="avg",
        shots=1,
        interval=100.0,
        dsp=DspConfig(
            enable_dsp_demodulation=True,
            enable_dsp_sum=False,
            enable_dsp_classification=False,
            line_param0=(1.0, 0.0, 0.0),
            line_param1=(0.0, 1.0, 0.0),
        ),
        frequency=FrequencyConfig(frequencies={}),
    )

    schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q01"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    adapter.build_execution_request(schedule=schedule, config=config)

    assert backend_controller.targets == ["Q01", "Q00", "Q02"]
