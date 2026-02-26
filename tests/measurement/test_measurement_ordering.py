"""Tests for ordering behavior in measurement internals."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

from qxpulse import FlatTop, PulseSchedule

from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.system import ControlParams, Target


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


def test_schedule_builder_uses_registry_for_readout_label_order(
    monkeypatch,
) -> None:
    """Given custom labels, when registry is present, then readout order follows registry mapping."""
    target_map = {
        "custom-1": "RQ01",
        "custom-0": "RQ00",
    }

    class _TargetRegistry:
        @staticmethod
        def resolve_read_label(label: str) -> str:
            return target_map[label]

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
                "custom-1": SimpleNamespace(is_pump=False, is_read=False),
                "custom-0": SimpleNamespace(is_pump=False, is_read=False),
            },
        ),
        mux_dict={},
        target_registry=cast(Any, _TargetRegistry()),
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

    with PulseSchedule(["custom-1", "custom-0", "custom-1"]) as schedule:
        pass

    builder.build(schedule=schedule, add_last_measurement=True)

    assert captured["readout_targets"] == ["RQ01", "RQ00"]


def test_schedule_builder_accepts_qubit_keyed_readout_amplitudes(monkeypatch) -> None:
    """Given qubit-keyed amplitudes, when adding readout, then mapped readout targets receive them."""
    captured: dict[str, Any] = {}

    def _readout_pulse(**kwargs: Any) -> FlatTop:
        captured["target"] = kwargs["target"]
        captured["amplitude"] = kwargs["amplitude"]
        return FlatTop(duration=16, amplitude=0.1, tau=4)

    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParams,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
        ),
        pulse_factory=cast(
            MeasurementPulseFactory,
            SimpleNamespace(readout_pulse=_readout_pulse),
        ),
        targets=cast(
            dict[str, Target],
            {
                "Q00": SimpleNamespace(is_pump=False, is_read=False),
            },
        ),
        mux_dict={},
    )

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(
            lambda self, *, schedule, readout_targets: CaptureSchedule(captures=[]),
            builder,
        ),
    )

    with PulseSchedule(["Q00"]) as schedule:
        pass

    builder.build(
        schedule=schedule,
        add_last_measurement=True,
        readout_amplitudes={"Q00": 0.37},
    )

    assert captured["target"] == "RQ00"
    assert captured["amplitude"] == 0.37


def test_backend_adapter_keeps_target_merge_order(monkeypatch) -> None:
    """Given gen and cap targets, when building request, then target merge order is deterministic."""

    class _BackendController:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def get_resource_map(self, targets: list[str]) -> dict[str, Any]:
            self.targets = targets
            return {}

    backend_controller = _BackendController()
    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend_controller),
            experiment_system=cast(Any, object()),
        ),
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
        n_shots=1,
        shot_interval_ns=100.0,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )

    schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["Q01"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    adapter.build_execution_request(schedule=schedule, config=config)

    assert backend_controller.targets == ["Q01", "Q00", "Q02"]
