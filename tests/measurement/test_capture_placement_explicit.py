"""Tests for explicit capture placement."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from qxpulse import Blank, FlatTop, PulseSchedule

from qubex.backend.quel1 import SAMPLING_PERIOD_NS
from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.system import ControlParameters, Target

QUEL1 = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)
QUEL3 = MeasurementConstraintProfile.quel3(sampling_period_ns=SAMPLING_PERIOD_NS)
BLOCK_DURATION = cast(float, QUEL1.block_duration_ns)
WORD_DURATION = cast(float, QUEL1.word_duration_ns)
LEADING_PAD = QUEL1.extra_capture_duration_ns
WORKAROUND_DURATION = QUEL1.workaround_capture_duration_ns

READOUT = "RQ00"
OTHER = "RQ01"


def make_builder(
    profile: MeasurementConstraintProfile = QUEL1,
) -> MeasurementScheduleBuilder:
    """Return a builder wired with stub system state."""
    return MeasurementScheduleBuilder(
        control_params=cast(
            ControlParameters,
            SimpleNamespace(readout_amplitude={READOUT: 0.1, OTHER: 0.1}),
        ),
        pulse_factory=cast(
            MeasurementPulseFactory,
            SimpleNamespace(
                readout_pulse=lambda **_: FlatTop(duration=64, amplitude=0.1, tau=0)
            ),
        ),
        targets=cast(
            dict[str, Target],
            {
                READOUT: SimpleNamespace(is_pump=False, is_read=True),
                OTHER: SimpleNamespace(is_pump=False, is_read=True),
            },
        ),
        mux_dict={},
        constraint_profile=profile,
    )


def make_schedule(*, pulse_duration: float = 64.0, gap: float = 0.0) -> PulseSchedule:
    """Return a readout-only schedule holding one or two readout pulses."""
    schedule = PulseSchedule([READOUT])
    with schedule as sequence:
        sequence.add(READOUT, FlatTop(duration=pulse_duration, amplitude=0.1, tau=0))
        if gap > 0.0:
            sequence.add(READOUT, Blank(duration=gap))
            sequence.add(
                READOUT, FlatTop(duration=pulse_duration, amplitude=0.1, tau=0)
            )
    return schedule


def data_windows(
    schedule: MeasurementSchedule, channel: str = READOUT
) -> list[Capture]:
    """Return non-workaround captures on `channel` ordered by start time."""
    captures = [
        c
        for c in schedule.capture_schedule.captures
        if not c.is_workaround and channel in c.channels
    ]
    return sorted(captures, key=lambda capture: capture.start_time)


def build_explicit(
    schedule: PulseSchedule,
    capture_windows: dict[str, list[tuple[float, float]]] | None,
    *,
    profile: MeasurementConstraintProfile = QUEL1,
) -> MeasurementSchedule:
    """Build `schedule` with explicit capture placement."""
    return make_builder(profile).build(
        schedule=schedule,
        capture_placement="explicit",
        capture_windows=capture_windows,
        final_measurement=False,
    )


def test_capture_schedule_placement_defaults_to_pulse_aligned() -> None:
    """CaptureSchedule should default to pulse-aligned placement."""
    assert CaptureSchedule(captures=[]).placement == "pulse_aligned"


def test_explicit_window_is_shifted_by_quel1_leading_pad() -> None:
    """Given a window at the schedule start, when building for QuEL-1, then it lands after the leading pad."""
    result = build_explicit(make_schedule(), {READOUT: [(0.0, 64.0)]})

    windows = data_windows(result)
    assert len(windows) == 1
    assert windows[0].start_time == LEADING_PAD
    assert windows[0].duration == 64.0
    assert result.capture_schedule.placement == "explicit"


def test_explicit_placement_keeps_workaround_capture() -> None:
    """Explicit placement should still emit the leading workaround capture on QuEL-1."""
    result = build_explicit(make_schedule(), {READOUT: [(0.0, 64.0)]})

    first = result.capture_schedule.captures[0]
    assert first.is_workaround is True
    assert first.start_time == 0.0
    assert first.duration == WORKAROUND_DURATION


def test_explicit_allows_several_windows_per_readout() -> None:
    """Given one readout and two windows, when building, then both windows are emitted."""
    result = build_explicit(
        make_schedule(pulse_duration=128.0),
        {READOUT: [(0.0, 64.0), (96.0, 64.0)]},
    )

    windows = data_windows(result)
    assert [(w.start_time, w.duration) for w in windows] == [
        (LEADING_PAD, 64.0),
        (LEADING_PAD + 96.0, 64.0),
    ]


def test_explicit_allows_one_window_across_two_readouts() -> None:
    """Given two readouts and one spanning window, when building, then a single window is emitted."""
    result = build_explicit(make_schedule(gap=128.0), {READOUT: [(0.0, 256.0)]})

    windows = data_windows(result)
    assert [(w.start_time, w.duration) for w in windows] == [(LEADING_PAD, 256.0)]


def test_explicit_allows_window_on_channel_without_readout_pulse() -> None:
    """Given a blank-only channel, when a window is requested on it, then it is captured."""
    schedule = PulseSchedule([READOUT, OTHER])
    with schedule as sequence:
        sequence.add(READOUT, FlatTop(duration=64, amplitude=0.1, tau=0))
        sequence.add(OTHER, Blank(duration=256))

    result = build_explicit(schedule, {OTHER: [(64.0, 128.0)]})

    assert [(w.start_time, w.duration) for w in data_windows(result, OTHER)] == [
        (LEADING_PAD + 64.0, 128.0)
    ]
    assert data_windows(result, READOUT) == []


def test_explicit_rejects_channel_missing_from_schedule() -> None:
    """Given a window on a label absent from the schedule, when building, then it raises ValueError."""
    with pytest.raises(ValueError, match="RQ99"):
        build_explicit(make_schedule(), {"RQ99": [(0.0, 64.0)]})


@pytest.mark.parametrize("capture_windows", [None, {}, {READOUT: []}])
def test_explicit_requires_capture_windows(
    capture_windows: dict[str, list[tuple[float, float]]] | None,
) -> None:
    """Given explicit placement without windows, when building, then it raises ValueError."""
    with pytest.raises(ValueError, match="capture_windows"):
        build_explicit(make_schedule(), capture_windows)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ((0.0, 0.0), "duration must be a finite, positive"),
        ((0.0, -8.0), "duration must be a finite, positive"),
        ((0.0, float("inf")), "duration must be a finite, positive"),
        ((-8.0, 64.0), "start must be a finite, non-negative"),
        ((float("nan"), 64.0), "start must be a finite, non-negative"),
    ],
)
def test_explicit_rejects_invalid_window(
    window: tuple[float, float], message: str
) -> None:
    """Given a non-finite, negative, or empty window, when building, then it raises naming channel, window and rule."""
    with pytest.raises(
        ValueError, match=rf"window 0 \({window[0]}, {window[1]}\) on `RQ00`.*{message}"
    ):
        build_explicit(make_schedule(), {READOUT: [window]})


def test_explicit_rejects_start_off_quel1_word_grid() -> None:
    """Given a 7 ns start on QuEL-1, when building, then it raises instead of flooring to 0 ns."""
    with pytest.raises(
        ValueError,
        match=r"window 0 \(7.0, 32.0\) on `RQ00`: start must be a multiple of 8.0 ns",
    ):
        build_explicit(make_schedule(), {READOUT: [(7.0, 32.0)]})


def test_explicit_rejects_duration_off_quel1_word_grid() -> None:
    """Given a 36 ns duration on QuEL-1, when building, then it raises instead of rounding."""
    with pytest.raises(
        ValueError,
        match=r"window 1 \(96.0, 36.0\) on `RQ00`: duration must be a multiple of 8.0 ns",
    ):
        build_explicit(make_schedule(), {READOUT: [(0.0, 32.0), (96.0, 36.0)]})


def test_explicit_uses_on_grid_window_unchanged() -> None:
    """Given an on-grid window, when building for QuEL-1, then only the leading pad is added."""
    result = build_explicit(make_schedule(), {READOUT: [(8.0, 32.0)]})

    window = data_windows(result)[0]
    assert (window.start_time, window.duration) == (LEADING_PAD + 8.0, 32.0)


def test_explicit_extends_schedule_to_fit_window() -> None:
    """Given a window longer than the pulse schedule, when building, then the schedule is extended."""
    result = build_explicit(make_schedule(), {READOUT: [(0.0, 1024.0)]})

    window = data_windows(result)[0]
    assert window.start_time + window.duration <= result.pulse_schedule.duration
    assert result.pulse_schedule.duration % BLOCK_DURATION == 0.0


@pytest.mark.parametrize(
    "windows",
    [
        [(0.0, 128.0), (64.0, 64.0)],  # overlap
        [(0.0, 64.0), (64.0, 64.0)],  # no post-blank
    ],
)
def test_explicit_rejects_windows_too_close_on_quel1(
    windows: list[tuple[float, float]],
) -> None:
    """Given windows that overlap or touch, when building for QuEL-1, then it raises naming both windows."""
    with pytest.raises(
        ValueError,
        match=r"\(0.0, \d+.0\) and \(64.0, 64.0\) on `RQ00` overlap or leave less than 8.0 ns",
    ):
        build_explicit(make_schedule(), {READOUT: windows})


def test_explicit_passes_windows_through_on_quel3() -> None:
    """Given off-grid windows, when building for QuEL-3, then they are used as given without a workaround."""
    result = build_explicit(
        make_schedule(),
        {READOUT: [(6.0, 40.0), (300.0, 50.0)]},
        profile=QUEL3,
    )

    assert all(not c.is_workaround for c in result.capture_schedule.captures)
    assert [(w.start_time, w.duration) for w in data_windows(result)] == [
        (6.0, 40.0),
        (300.0, 50.0),
    ]
    assert result.pulse_schedule.duration == 350.0


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


def make_validator_schedule(
    *,
    placement: str,
    window_start: float,
    window_duration: float,
) -> MeasurementSchedule:
    """Return a schedule whose window does not match the readout pulse range."""
    pulse_schedule = _FakePulseSchedule(
        duration=4 * BLOCK_DURATION,
        length=round((4 * BLOCK_DURATION) / SAMPLING_PERIOD_NS),
        ranges={READOUT: [_FakeRange(start=20, stop=52)]},
    )
    capture_schedule = CaptureSchedule(
        captures=[
            Capture(
                channels=[READOUT],
                start_time=0.0,
                duration=WORKAROUND_DURATION,
                is_workaround=True,
            ),
            Capture(
                channels=[READOUT],
                start_time=window_start,
                duration=window_duration,
            ),
        ],
        placement=cast("str", placement),  # type: ignore[arg-type]
    )
    return MeasurementSchedule.model_construct(
        pulse_schedule=pulse_schedule,
        capture_schedule=capture_schedule,
    )


def make_adapter() -> Quel1MeasurementBackendAdapter:
    """Return a QuEL-1 adapter whose validator needs no backend."""
    return Quel1MeasurementBackendAdapter(
        backend_controller=object(),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
    )


def test_validator_accepts_explicit_windows_detached_from_pulses() -> None:
    """Given explicit placement, when the window does not match a pulse range, then validation passes."""
    schedule = make_validator_schedule(
        placement="explicit",
        window_start=LEADING_PAD,
        window_duration=128.0,
    )

    assert make_adapter().validate_schedule(schedule) is None


def test_validator_rejects_detached_windows_for_pulse_aligned_placement() -> None:
    """Given pulse-aligned placement, when the window does not match a pulse range, then it raises."""
    schedule = make_validator_schedule(
        placement="pulse_aligned",
        window_start=LEADING_PAD,
        window_duration=128.0,
    )

    with pytest.raises(ValueError, match="mismatch for RQ00"):
        make_adapter().validate_schedule(schedule)


def test_validator_still_enforces_word_grid_for_explicit_placement() -> None:
    """Given explicit placement, when the window is off the word grid, then it raises ValueError."""
    schedule = make_validator_schedule(
        placement="explicit",
        window_start=LEADING_PAD + 2.0,
        window_duration=128.0,
    )

    with pytest.raises(ValueError, match="multiple of"):
        make_adapter().validate_schedule(schedule)


def make_config() -> MeasurementConfig:
    """Return a minimal measurement configuration."""
    return MeasurementConfig(
        n_shots=1,
        shot_interval=1000,
        shot_averaging=True,
        time_integration=True,
        state_classification=False,
    )


class _RecordingBackend:
    """Backend stub that records execute calls and never reaches hardware."""

    sampling_period_ns = SAMPLING_PERIOD_NS
    CAPTURE_DECIMATION_FACTOR = 4

    def __init__(self) -> None:
        self.execute_calls: list[object] = []

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict[str, str]]]:
        return {target: [{}] for target in targets}

    def execute_sync(self, **kwargs: object) -> None:
        self.execute_calls.append(kwargs)


def make_adapter_with_backend() -> tuple[
    Quel1MeasurementBackendAdapter, _RecordingBackend
]:
    """Return a QuEL-1 adapter bound to a recording backend stub."""
    backend = _RecordingBackend()
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast("object", backend),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
    )
    return adapter, backend


def test_build_request_rejects_explicit_duration_off_decimation_grid() -> None:
    """Given demodulation on and a 40 ns explicit window, when building the request, then it raises ValueError."""
    adapter, _ = make_adapter_with_backend()
    schedule = make_validator_schedule(
        placement="explicit",
        window_start=LEADING_PAD,
        window_duration=40.0,
    )

    with pytest.raises(
        ValueError,
        match=r"\(40.0, 40.0\) on `RQ00`: duration must be a multiple of 32.0 ns while demodulation",
    ):
        adapter.build_execution_request(schedule=schedule, config=make_config())


def test_build_request_accepts_explicit_duration_without_demodulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given demodulation off and a 40 ns explicit window, when building the request, then no grid error is raised."""
    adapter, _ = make_adapter_with_backend()
    monkeypatch.setattr(
        adapter,
        "_create_sampled_sequences",
        lambda *, schedule: ({}, {}),
        raising=False,
    )
    schedule = make_validator_schedule(
        placement="explicit",
        window_start=LEADING_PAD,
        window_duration=40.0,
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=make_config(),
        quel1_options=Quel1MeasurementOptions(demodulation=False),
    )

    assert request.payload.dsp_demodulation is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("window_start", "window_duration", "message"),
    [
        (LEADING_PAD + 2.0, 128.0, "multiple of 8.0 ns"),  # validate_schedule
        (LEADING_PAD, 40.0, "multiple of 32.0 ns while demodulation"),  # request build
    ],
)
def test_runner_does_not_touch_backend_when_explicit_window_is_invalid(
    window_start: float, window_duration: float, message: str
) -> None:
    """Given an invalid explicit window, when executing, then it raises and the backend is never called."""
    adapter, backend = make_adapter_with_backend()
    runner = MeasurementScheduleRunner(
        measurement_backend_adapter=adapter,
        backend_controller=cast("object", backend),  # type: ignore[arg-type]
    )
    schedule = make_validator_schedule(
        placement="explicit",
        window_start=window_start,
        window_duration=window_duration,
    )

    with pytest.raises(ValueError, match=message):
        runner.execute_sync(schedule=schedule, config=make_config())
    assert backend.execute_calls == []
