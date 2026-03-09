"""Tests for ordering behavior in measurement internals."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
from qxpulse import FlatTop, PulseSchedule

from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.system import ControlParameters, Target


def test_schedule_builder_keeps_readout_target_order(monkeypatch) -> None:
    """Given schedule labels, when adding readout targets, then first-seen label order is kept."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
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
        capture_targets: list[str],
        capture_placement: str,
    ) -> CaptureSchedule:
        _ = (capture_targets, capture_placement)
        captured["readout_targets"] = readout_targets
        return CaptureSchedule(captures=[])

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(_capture, builder),
    )

    with PulseSchedule(["Q01", "Q00", "Q01"]) as schedule:
        pass

    builder.build(schedule=schedule, final_measurement=True)

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
            ControlParameters,
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
        capture_targets: list[str],
        capture_placement: str,
    ) -> CaptureSchedule:
        _ = (capture_targets, capture_placement)
        captured["readout_targets"] = readout_targets
        return CaptureSchedule(captures=[])

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(_capture, builder),
    )

    with PulseSchedule(["custom-1", "custom-0", "custom-1"]) as schedule:
        pass

    builder.build(schedule=schedule, final_measurement=True)

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
            ControlParameters,
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

    def _capture_noop(
        self: MeasurementScheduleBuilder,
        *,
        schedule: PulseSchedule,
        readout_targets: list[str],
        capture_targets: list[str],
        capture_placement: str,
    ) -> CaptureSchedule:
        _ = (schedule, readout_targets, capture_targets, capture_placement)
        return CaptureSchedule(captures=[])

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(_capture_noop, builder),
    )

    with PulseSchedule(["Q00"]) as schedule:
        pass

    builder.build(
        schedule=schedule,
        final_measurement=True,
        readout_amplitudes={"Q00": 0.37},
    )

    assert captured["target"] == "RQ00"
    assert captured["amplitude"] == 0.37


def test_schedule_builder_resolves_none_options_in_build(monkeypatch) -> None:
    """Given None build options, when building, then builder applies internal defaults."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
        ),
        pulse_factory=cast(MeasurementPulseFactory, SimpleNamespace()),
        targets=cast(
            dict[str, Target],
            {
                "RQ00": SimpleNamespace(is_pump=False, is_read=True),
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
        capture_targets: list[str],
        capture_placement: str,
    ) -> CaptureSchedule:
        _ = (self, schedule, capture_targets)
        captured["readout_targets"] = readout_targets
        captured["capture_placement"] = capture_placement
        return CaptureSchedule(captures=[])

    monkeypatch.setattr(
        builder,
        "_build_capture_schedule",
        MethodType(_capture, builder),
    )

    with PulseSchedule(["RQ00"]) as schedule:
        pass

    builder.build(
        schedule=schedule,
        readout_amplification=None,
        final_measurement=None,
        capture_placement=None,
        plot=None,
    )

    assert captured["readout_targets"] == ["RQ00"]
    assert captured["capture_placement"] == "pulse_aligned"


def test_schedule_builder_applies_frequency_overrides_to_existing_channels() -> None:
    """Given frequency overrides, when building without final measurement, then channel frequencies are updated."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
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
                "Q00": SimpleNamespace(is_pump=False, is_read=False),
                "RQ00": SimpleNamespace(is_pump=False, is_read=True),
            },
        ),
        mux_dict={},
    )

    with PulseSchedule(["Q00", "RQ00"]) as schedule:
        pass

    result = builder.build(
        schedule=schedule,
        frequencies={"Q00": 5.1, "RQ00": 9.8},
    )

    assert result.pulse_schedule.get_frequency("Q00") == 5.1
    assert result.pulse_schedule.get_frequency("RQ00") == 9.8


def test_schedule_builder_keeps_frequency_overrides_after_final_measurement() -> None:
    """Given frequency overrides, when appending final measurement, then added readout channels keep frequencies."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
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
                "Q00": SimpleNamespace(is_pump=False, is_read=False),
            },
        ),
        mux_dict={},
    )

    with PulseSchedule(["Q00"]) as schedule:
        pass

    result = builder.build(
        schedule=schedule,
        final_measurement=True,
        frequencies={"Q00": 5.2, "RQ00": 9.9},
    )

    assert result.pulse_schedule.get_frequency("Q00") == 5.2
    assert result.pulse_schedule.get_frequency("RQ00") == 9.9


def test_schedule_builder_rejects_unknown_frequency_override_targets() -> None:
    """Given unknown frequency labels, when building, then frequency overrides are rejected."""
    builder = MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={"RQ00": 0.1}),
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
                "Q00": SimpleNamespace(is_pump=False, is_read=False),
            },
        ),
        mux_dict={},
    )

    with PulseSchedule(["Q00"]) as schedule:
        pass

    with pytest.raises(ValueError, match="Unknown frequency override target"):
        builder.build(
            schedule=schedule,
            final_measurement=True,
            frequencies={"Q99": 5.4},
        )


def test_backend_adapter_keeps_target_merge_order(monkeypatch) -> None:
    """Given gen and cap targets, when building request, then target merge order is deterministic."""

    class _BackendController:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def get_resource_map(self, targets: list[str]) -> dict[str, Any]:
            self.targets = targets
            return {target: {} for target in targets}

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
        shot_interval=100.0,
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
