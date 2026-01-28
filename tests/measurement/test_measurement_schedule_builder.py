"""Tests for MeasurementScheduleBuilder helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from qubex.backend import SAMPLING_PERIOD, ControlParams, Mux, Target
from qubex.backend.quel_instrument_executor import (
    EXTRA_SUM_SECTION_LENGTH,
    WORD_LENGTH,
)
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.pulse import Blank, FlatTop, PulseArray, PulseSchedule
from qubex.typing import TargetMap


class StubReadoutFactory:
    """Stub factory that records readout pulse calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        """Record call arguments and return a dummy pulse array."""
        self.calls.append(kwargs)
        return PulseArray([Blank(4)])


class StubPumpFactory:
    """Stub factory that records pump pulse calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        """Record call arguments and return a dummy flat-top pulse."""
        self.calls.append(kwargs)
        return FlatTop(duration=4, amplitude=0.0, tau=1.0)


def test_add_readout_pulses_adds_readout_channel() -> None:
    """Ensure readout pulses create a readout channel in the schedule."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {
            "Q00": SimpleNamespace(is_pump=False, is_read=False),
            "RQ00": SimpleNamespace(is_pump=False, is_read=True),
        },
    )
    control_params = cast(
        ControlParams,
        SimpleNamespace(readout_amplitude={"RQ00": 0.2}),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict={},
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule(["Q00"])
    builder.add_readout_pulses(
        schedule=schedule,
        readout_amplitudes=None,
        readout_duration=12,
        readout_pre_margin=1,
        readout_post_margin=2,
        readout_ramptime=3,
        readout_drag_coeff=0.1,
        readout_ramp_type="RaisedCosine",
    )

    assert "RQ00" in schedule.labels
    assert readout_factory.calls == [
        {
            "target": "RQ00",
            "duration": 12,
            "amplitude": 0.2,
            "ramptime": 3,
            "type": "RaisedCosine",
            "drag_coeff": 0.1,
            "pre_margin": 1,
            "post_margin": 2,
        }
    ]


def test_add_pump_pulses_adds_mux_channel() -> None:
    """Ensure pump pulses create a mux channel in the schedule."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {"RQ00": SimpleNamespace(is_pump=False, is_read=True)},
    )
    mux = cast(Mux, SimpleNamespace(index=0, label="MX0"))
    control_params = cast(
        ControlParams,
        SimpleNamespace(get_pump_amplitude=lambda _: 0.3),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict=cast(dict[str, Mux], {"Q00": mux}),
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule([])
    builder.add_pump_pulses(
        schedule=schedule,
        readout_ranges={"RQ00": [range(4, 8)]},
        readout_pre_margin=0,
        readout_ramptime=2,
        readout_ramp_type="RaisedCosine",
    )

    assert "MX0" in schedule.labels
    assert pump_factory.calls[0]["target"] == "RQ00"
    assert pump_factory.calls[0]["amplitude"] == 0.3


def test_build_measurement_schedule_requires_readout_targets() -> None:
    """Given no readout targets, when building, then raise a ValueError."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {"Q00": SimpleNamespace(is_pump=False, is_read=False)},
    )
    control_params = cast(
        ControlParams,
        SimpleNamespace(readout_amplitude={}, capture_delay_word={}),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict={},
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule(["Q00"])
    schedule.add("Q00", Blank(4))

    with pytest.raises(ValueError, match="No readout targets"):
        builder.build_measurement_schedule(
            schedule=schedule,
            add_last_measurement=False,
            add_pump_pulses=False,
        )


def test_create_capture_schedule_rejects_misaligned_ranges() -> None:
    """Given misaligned range, when capturing, then raise a ValueError."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {"RQ00": SimpleNamespace(is_pump=False, is_read=True)},
    )
    mux = cast(Mux, SimpleNamespace(index=0, label="MX0"))
    control_params = cast(
        ControlParams,
        SimpleNamespace(capture_delay_word={}),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict=cast(dict[str, Mux], {"Q00": mux}),
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule(["RQ00"])
    schedule.add("RQ00", Blank(8))

    with pytest.raises(ValueError, match="Capture range should start"):
        builder.create_capture_schedule(
            schedule=schedule,
            readout_ranges={"RQ00": [range(2, 6)]},
        )


def test_create_capture_schedule_constructs_expected_captures() -> None:
    """Ensure capture schedule contains extra-sum and per-range captures."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {"RQ00": SimpleNamespace(is_pump=False, is_read=True)},
    )
    mux = cast(Mux, SimpleNamespace(index=1, label="MX1"))
    control_params = cast(
        ControlParams,
        SimpleNamespace(capture_delay_word={1: 2}),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict=cast(dict[str, Mux], {"Q00": mux}),
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule(["RQ00"])
    schedule.add("RQ00", Blank(duration=100))

    first_range = range(0, WORD_LENGTH)
    second_range = range(WORD_LENGTH * 2, WORD_LENGTH * 3)
    capture_schedule = builder.create_capture_schedule(
        schedule=schedule,
        readout_ranges={"RQ00": [first_range, second_range]},
    )

    captures = capture_schedule.captures
    assert len(captures) == 3

    delay_samples = 2 * WORD_LENGTH
    delay_time = delay_samples * SAMPLING_PERIOD
    assert captures[0].channels == ["RQ00"]
    assert captures[0].start_time == delay_time
    assert captures[0].duration == EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD

    first_capture = captures[1]
    assert first_capture.start_time == delay_time
    assert first_capture.duration == len(first_range) * SAMPLING_PERIOD

    second_capture = captures[2]
    assert second_capture.start_time == (
        (second_range.start + delay_samples) * SAMPLING_PERIOD
    )
    assert second_capture.duration == len(second_range) * SAMPLING_PERIOD


def test_create_capture_schedule_rejects_consecutive_ranges_without_blank() -> None:
    """Reject capture creation when readout pulses lack required post blank."""
    readout_factory = StubReadoutFactory()
    pump_factory = StubPumpFactory()

    targets = cast(
        TargetMap[Target],
        {"RQ00": SimpleNamespace(is_pump=False, is_read=True)},
    )
    mux = cast(Mux, SimpleNamespace(index=0, label="MX0"))
    control_params = cast(
        ControlParams,
        SimpleNamespace(capture_delay_word={}),
    )

    builder = MeasurementScheduleBuilder(
        targets=targets,
        mux_dict=cast(dict[str, Mux], {"Q00": mux}),
        control_params=control_params,
        readout_pulse_factory=readout_factory,
        pump_pulse_factory=pump_factory,
    )

    schedule = PulseSchedule(["RQ00"])
    schedule.add("RQ00", Blank(duration=100))

    with pytest.raises(ValueError, match="post-blank time"):
        builder.create_capture_schedule(
            schedule=schedule,
            readout_ranges={
                "RQ00": [
                    range(0, WORD_LENGTH),
                    range(WORD_LENGTH, WORD_LENGTH * 2),
                ]
            },
        )
