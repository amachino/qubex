"""Regression tests for explicit capture placement and capture-delay routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule, Rect

from qubex.backend.quel1.compat.driver_loader import load_quel1_driver
from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import (
    CapturePlacement,
    MeasurementScheduleBuilder,
)
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.system import ControlParameters, Target


def _builder(profile: MeasurementConstraintProfile) -> MeasurementScheduleBuilder:
    return MeasurementScheduleBuilder(
        control_params=cast(ControlParameters, SimpleNamespace(readout_amplitude={})),
        pulse_factory=cast(MeasurementPulseFactory, SimpleNamespace()),
        targets=cast(
            dict[str, Target], {"RQ00": SimpleNamespace(is_read=True, is_pump=False)}
        ),
        mux_dict={},
        constraint_profile=profile,
    )


def _readout(leading_blank: int = 0) -> PulseSchedule:
    schedule = PulseSchedule(["RQ00"])
    if leading_blank:
        schedule.add("RQ00", Blank(leading_blank))
    schedule.add("RQ00", Rect(duration=160, amplitude=0.1))
    return schedule


def _adapter_and_backend() -> tuple[Quel1MeasurementBackendAdapter, Any]:
    backend = SimpleNamespace(
        driver=load_quel1_driver(),
        get_resource_map=lambda targets: {target: [{}] for target in targets},
    )
    system = SimpleNamespace(
        control_params=SimpleNamespace(capture_delay_word={0: 16}),
        get_mux_by_qubit=lambda _: SimpleNamespace(index=0),
        get_target=lambda _: SimpleNamespace(sideband="U"),
        get_diff_frequency=lambda _: 0.01,
        get_awg_frequency=lambda _: 0.01,
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, backend),
        experiment_system=cast(Any, system),
        constraint_profile=MeasurementConstraintProfile.quel1(),
    )
    return adapter, backend


@pytest.mark.parametrize("placement", ["pulse_aligned", "entire_schedule"])
@pytest.mark.parametrize("strict", [True, False])
def test_builder_records_explicit_placement(
    placement: CapturePlacement, strict: bool
) -> None:
    """The builder records the requested placement on strict and relaxed measurement schedules."""
    profile = (
        MeasurementConstraintProfile.quel1()
        if strict
        else MeasurementConstraintProfile.quel3(2.0)
    )
    schedule = _builder(profile).build(schedule=_readout(), capture_placement=placement)

    assert schedule.capture_placement == placement
    assert schedule.model_copy(deep=True).capture_placement == placement
    assert (
        MeasurementSchedule.model_validate(schedule.model_dump()).capture_placement
        == placement
    )


def test_manual_schedule_defaults_to_pulse_aligned() -> None:
    """Existing manually constructed schedules retain ordinary measurement as their default placement."""
    schedule = MeasurementSchedule(
        pulse_schedule=PulseSchedule(["RQ00"]),
        capture_schedule=CaptureSchedule(captures=[]),
    )

    assert schedule.capture_placement == "pulse_aligned"


def test_colliding_capture_geometry_preserves_normal_delay_and_phase() -> None:
    """Identical capture geometry must not erase the ordinary capture delay or its phase compensation."""
    builder = _builder(MeasurementConstraintProfile.quel1())
    ordinary = builder.build(schedule=_readout(24), capture_placement="pulse_aligned")
    entire = builder.build(schedule=_readout(), capture_placement="entire_schedule")
    assert ordinary.pulse_schedule.duration == entire.pulse_schedule.duration == 256
    assert ordinary.capture_schedule == entire.capture_schedule
    assert ordinary.capture_schedule.captures[1].start_time == 64
    assert ordinary.capture_schedule.captures[1].duration == 160
    config = MeasurementConfig(
        n_shots=64,
        shot_interval=200_000,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    adapter, backend = _adapter_and_backend()
    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=adapter,
        backend_controller=backend,
        constraint_profile=MeasurementConstraintProfile.quel1(),
    )

    for schedule, expected_delay in [(ordinary, 64), (entire, 0)]:
        request = runner._prepare_execution(schedule=schedule, config=config)  # noqa: SLF001
        payload = cast(Any, request.payload)
        cap_sequence = payload.cap_sampled_sequence["RQ00"]
        assert cap_sequence.sub_sequences[0].prev_blank == expected_delay
        param = backend.driver.CaptureParamTools.create(
            sequence=cap_sequence,
            capture_delay_words=0,
            repeats=config.n_shots,
            interval_samples=payload.interval_ns // 2,
        )
        assert param.capture_delay == expected_delay // 4
        generated = payload.gen_sampled_sequence["RQ00"].sub_sequences[0]
        waveform = np.asarray(generated.real) + 1j * np.asarray(generated.imag)
        expected_phase = np.exp(-2j * np.pi * 0.01 * (64 + expected_delay * 2))
        np.testing.assert_allclose(
            waveform[32:112], np.full(80, 0.1 * expected_phase), rtol=0, atol=1e-14
        )


def test_full_span_geometry_does_not_bypass_normal_capture_validation() -> None:
    """Ordinary placement validates pulse alignment even if its windows resemble full-span capture."""
    full = _builder(MeasurementConstraintProfile.quel1()).build(
        schedule=_readout(24), capture_placement="entire_schedule"
    )
    ordinary = MeasurementSchedule(
        pulse_schedule=full.pulse_schedule,
        capture_schedule=full.capture_schedule,
    )
    adapter, _ = _adapter_and_backend()

    with pytest.raises(ValueError, match="Capture start mismatch"):
        adapter.validate_schedule(ordinary)
