"""Tests for measurement schedule constraint validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.backend.quel1 import SAMPLING_PERIOD
from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_schedule_builder import (
    BLOCK_DURATION,
    EXTRA_SUM_SECTION_LENGTH,
    WORD_DURATION,
)
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_schedule import MeasurementSchedule


@dataclass
class _FakeRange:
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass
class _FakePulseSchedule:
    duration: float
    length: int
    ranges: dict[str, list[_FakeRange]]

    def is_valid(self) -> bool:
        return True

    def get_pulse_ranges(
        self, labels: list[str] | None = None
    ) -> dict[str, list[_FakeRange]]:
        if labels is None:
            return self.ranges
        return {label: self.ranges.get(label, []) for label in labels}


def test_validate_measurement_schedule_accepts_device_constraints() -> None:
    """Given aligned pulse/capture windows, when validating, then no error is raised."""
    target = "RQ00"
    pulse_schedule = _FakePulseSchedule(
        duration=2 * BLOCK_DURATION,
        length=round((2 * BLOCK_DURATION) / SAMPLING_PERIOD),
        ranges={target: [_FakeRange(start=64, stop=80)]},
    )
    capture_schedule = CaptureSchedule(
        captures=[
            Capture(
                channels=[target],
                start_time=0.0,
                duration=EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD,
            ),
            Capture(
                channels=[target],
                start_time=64 * SAMPLING_PERIOD,
                duration=16 * SAMPLING_PERIOD,
            ),
        ]
    )
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=pulse_schedule,
        capture_schedule=capture_schedule,
    )

    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=object(),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
    )
    adapter.validate_schedule(schedule)


def test_validate_measurement_schedule_rejects_non_block_aligned_first_capture() -> (
    None
):
    """Given first capture not block-aligned, when validating, then it raises ValueError."""
    target = "RQ00"
    pulse_schedule = _FakePulseSchedule(
        duration=2 * BLOCK_DURATION,
        length=round((2 * BLOCK_DURATION) / SAMPLING_PERIOD),
        ranges={target: [_FakeRange(start=64, stop=80)]},
    )
    capture_schedule = CaptureSchedule(
        captures=[
            Capture(
                channels=[target],
                start_time=WORD_DURATION,
                duration=EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD,
            ),
            Capture(
                channels=[target],
                start_time=64 * SAMPLING_PERIOD,
                duration=16 * SAMPLING_PERIOD,
            ),
        ]
    )
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=pulse_schedule,
        capture_schedule=capture_schedule,
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=object(),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="first capture start"):
        adapter.validate_schedule(schedule)


def test_validate_measurement_schedule_accepts_relaxed_profile() -> None:
    """Given relaxed constraints, when validating non-word/non-block schedule, then no error is raised."""
    target = "RQ00"
    pulse_schedule = _FakePulseSchedule(
        duration=123.0,
        length=62,
        ranges={target: [_FakeRange(start=5, stop=12)]},
    )
    capture_schedule = CaptureSchedule(
        captures=[
            Capture(
                channels=[target],
                start_time=10.0,
                duration=14.0,
            ),
        ]
    )
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=pulse_schedule,
        capture_schedule=capture_schedule,
    )

    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=object(),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
        constraint_profile=MeasurementConstraintProfile.quel3(2.0),
    )

    adapter.validate_schedule(schedule)
