"""Tests for CR dissipation schedule construction."""

# ruff: noqa: SLF001

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import qubex.contrib.experiment.cr_dissipation as health
from qubex.experiment import Experiment
from qubex.pulse import Blank, FlatTop, PulseSchedule, Rect, VirtualZ


class _DummyPulseService:
    """Provide deterministic pulses for schedule construction."""

    def x180(self, _target: str) -> Blank:
        """Return a four-nanosecond X180 stand-in."""
        return Blank(4.0)

    def y180(self, _target: str) -> Blank:
        """Return a four-nanosecond Y180 stand-in."""
        return Blank(4.0)

    def z180(self) -> VirtualZ:
        """Return a virtual Z180 pulse."""
        return VirtualZ(np.pi)


def _experiment() -> Experiment:
    """Return a minimal experiment-shaped object."""
    return cast(Experiment, SimpleNamespace(pulse=_DummyPulseService()))


def _schedule(duration: float) -> PulseSchedule:
    """Return a three-channel active ZX90 stand-in."""
    with PulseSchedule(["Q0", "Q0-Q1", "Q1"]) as schedule:
        schedule.add("Q0-Q1", Rect(duration=duration, amplitude=1.0))
    return schedule


def test_default_un_echoed_zx90_repeats_two_same_sign_lobes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default un-echoed ZX90 should concatenate two identical CR lobes."""
    primitive = _schedule(8.0)
    primitive.cr_duration = 8.0  # type: ignore[attr-defined]
    primitive.echo = False  # type: ignore[attr-defined]
    pulse_service = SimpleNamespace()
    monkeypatch.setattr(
        pulse_service,
        "zx90",
        lambda control, target, *, echo: primitive,
        raising=False,
    )
    exp = cast(Experiment, SimpleNamespace(pulse=pulse_service))

    gate = health._build_un_echoed_zx90(exp, "Q0", "Q1")

    assert gate.duration == pytest.approx(16.0)
    assert gate.cr_duration == pytest.approx(16.0)  # type: ignore[attr-defined]
    assert gate.echo is False  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        gate.get_sampled_sequence("Q0-Q1"),
        np.tile(primitive.get_sampled_sequence("Q0-Q1"), 2),
    )


def test_non_echoed_cr_unit_preserves_pi_sized_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protocol A/B units should place one blank after each positive CR lobe."""
    primitive = _schedule(8.0)
    primitive.cr_duration = 8.0  # type: ignore[attr-defined]
    primitive.echo = False  # type: ignore[attr-defined]
    pulse_service = SimpleNamespace()
    monkeypatch.setattr(
        pulse_service,
        "zx90",
        lambda control, target, *, echo: primitive,
        raising=False,
    )
    exp = cast(Experiment, SimpleNamespace(pulse=pulse_service))

    unit = health._build_un_echoed_zx90(exp, "Q0", "Q1", blank_duration=4.0)

    assert unit.duration == pytest.approx(24.0)
    assert unit.cr_duration == pytest.approx(16.0)  # type: ignore[attr-defined]


def test_ix45_initial_amplitude_uses_envelope_area_ratio() -> None:
    """The IX45 guess should convert calibrated X90 area to the CR envelope."""
    x90 = FlatTop(duration=40.0, amplitude=0.2, tau=10.0)
    cr = FlatTop(duration=100.0, amplitude=0.7, tau=20.0)

    amplitude = health._initial_cr_shaped_ix45_amplitude(x90, cr)

    expected = (
        0.5 * 0.2 * health._flat_top_unit_area(x90) / health._flat_top_unit_area(cr)
    )
    assert amplitude == pytest.approx(expected)
    assert amplitude != pytest.approx(0.1)


def test_cr_shaped_ix45_rejects_out_of_range_amplitude() -> None:
    """Reference amplitudes outside the hardware-normalized range are invalid."""
    cr = FlatTop(duration=100.0, amplitude=0.7, tau=20.0)

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        health._make_cr_shaped_ix45(cr, 1.1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        health._make_cr_shaped_ix45(cr, 0.0)

    zero_amplitude = health._make_cr_shaped_ix45(cr, 0.0, allow_zero=True)
    assert zero_amplitude.amplitude == 0.0


def test_cr_shaped_ix45_calibration_uses_calibration_note_timestamp_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached IX45 calibration timestamp must be readable by CalibrationNote."""
    target = "Q1"
    calls: list[dict[str, object]] = []

    def sweep_parameter(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        sequence = cast(Any, kwargs["sequence"])
        for amplitude in cast(np.ndarray, kwargs["sweep_range"]):
            sequence(float(amplitude))
        return SimpleNamespace(
            data={target: SimpleNamespace(normalized=np.linspace(-1.0, 1.0, 5))}
        )

    service = SimpleNamespace(sweep_parameter=sweep_parameter)
    exp = cast(Experiment, SimpleNamespace(measurement_service=service))
    cr = FlatTop(duration=100.0, amplitude=0.7, tau=20.0)
    monkeypatch.setattr(
        health.fitting,
        "fit_ampl_calib_data",
        lambda **kwargs: {"amplitude": 0.2, "r2": 0.9},
    )

    calibration = health._calibrate_cr_shaped_ix45(
        exp,
        target,
        cr,
        initial_amplitude=2.0,
        n_shots=100,
        shot_interval=1000.0,
        n_points=5,
    )

    datetime.strptime(calibration.timestamp, "%Y-%m-%d %H:%M:%S")
    np.testing.assert_allclose(calls[0]["sweep_range"], np.linspace(0.0, 1.0, 5))


def test_malformed_ix45_cache_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable cache entry should trigger recalibration instead of aborting."""
    cr = FlatTop(duration=100.0, amplitude=0.7, tau=20.0)
    calibrated = health.CrShapedIx45Calibration(
        amplitude=0.2,
        duration=100.0,
        ramptime=20.0,
        ramp_type=str(cr.type),
        sampling_period=cr.sampling_period,
        r_squared=0.9,
        timestamp="2026-09-14 12:00:00",
    )
    note = SimpleNamespace(
        get_property=lambda *args: (_ for _ in ()).throw(ValueError("bad timestamp")),
        put_property=lambda *args: None,
    )
    exp = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(calib_note=note),
            pulse=SimpleNamespace(get_hpi_pulse=lambda target: cr),
        ),
    )
    monkeypatch.setattr(
        health,
        "_calibrate_cr_shaped_ix45",
        lambda *args, **kwargs: calibrated,
    )

    with pytest.warns(RuntimeWarning, match="malformed cached"):
        result = health._resolve_cr_shaped_ix45_calibration(
            exp,
            "Q0",
            "Q1",
            cr,
            amplitude=None,
            valid_days=30,
            force=False,
            n_shots=100,
            shot_interval=1000.0,
            n_points=5,
            n_rotations=1,
            r2_threshold=0.5,
            plot=False,
        )

    assert result is calibrated


def test_missing_ix45_cache_category_triggers_first_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-created CalibrationNote category should behave as a cache miss."""
    cr = FlatTop(duration=100.0, amplitude=0.7, tau=20.0)
    calibrated = health.CrShapedIx45Calibration(
        amplitude=0.2,
        duration=100.0,
        ramptime=20.0,
        ramp_type=str(cr.type),
        sampling_period=cr.sampling_period,
        r_squared=0.9,
        timestamp="2026-09-14 12:00:00",
    )
    stored: list[tuple[object, ...]] = []
    note = SimpleNamespace(
        get_property=lambda *args: (_ for _ in ()).throw(
            AttributeError("'NoneType' object has no attribute 'get'")
        ),
        put_property=lambda *args: stored.append(args),
    )
    exp = cast(
        Experiment,
        SimpleNamespace(
            ctx=SimpleNamespace(calib_note=note),
            pulse=SimpleNamespace(get_hpi_pulse=lambda target: cr),
        ),
    )
    monkeypatch.setattr(
        health,
        "_calibrate_cr_shaped_ix45",
        lambda *args, **kwargs: calibrated,
    )

    result = health._resolve_cr_shaped_ix45_calibration(
        exp,
        "Q0",
        "Q1",
        cr,
        amplitude=None,
        valid_days=30,
        force=False,
        n_shots=100,
        shot_interval=1000.0,
        n_points=5,
        n_rotations=1,
        r2_threshold=0.5,
        plot=False,
    )

    assert result is calibrated
    assert len(stored) == 1


def test_echoed_reference_preserves_pi_slots_and_uses_two_ix45_lobes() -> None:
    """Echo references should replace only CR-active windows."""
    with PulseSchedule(["Q0", "Q0-Q1", "Q1"]) as echoed:
        echoed.add("Q0-Q1", Rect(duration=8.0, amplitude=1.0))
        echoed.barrier()
        echoed.add("Q0", Rect(duration=4.0, amplitude=1.0))
        echoed.barrier()
        echoed.add("Q0-Q1", Rect(duration=8.0, amplitude=-1.0))
        echoed.barrier()
        echoed.add("Q0", Rect(duration=4.0, amplitude=1.0))
    echoed.cr_duration = 8.0  # type: ignore[attr-defined]
    echoed.echo = True  # type: ignore[attr-defined]
    echoed.pi_pulse = Rect(duration=4.0, amplitude=1.0)  # type: ignore[attr-defined]
    ix45 = FlatTop(duration=8.0, amplitude=0.1, tau=2.0)

    reference = health._build_echoed_reference_unit(echoed, "Q0", "Q1", ix45)

    assert reference.duration == pytest.approx(echoed.duration)
    target_elements = reference.get_sequences(copy=True)["Q1"].flattened_elements
    assert sum(isinstance(item, FlatTop) for item in target_elements) == 2


def test_control_cr_transverse_echo_block_uses_documented_xy180_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control echo should apply XI, YI+IX, XI, then the closing YI."""
    exp = _experiment()
    pulses: list[tuple[str, str]] = []
    original_x = exp.pulse.x180
    original_y = exp.pulse.y180

    def record_x180(target: str) -> Blank:
        pulses.append(("X", target))
        return cast(Blank, original_x(target))

    def record_y180(target: str) -> Blank:
        pulses.append(("Y", target))
        return cast(Blank, original_y(target))

    monkeypatch.setattr(exp.pulse, "x180", record_x180)
    monkeypatch.setattr(exp.pulse, "y180", record_y180)

    block = health._control_cr_transverse_echo_block(exp, "Q0", "Q1", _schedule(20.0))

    assert pulses == [
        ("X", "Q0"),
        ("Y", "Q0"),
        ("X", "Q1"),
        ("X", "Q0"),
        ("Y", "Q0"),
    ]
    assert block.duration == pytest.approx(4 * 20.0 + 4 * 4.0)


def test_target_t2rho_block_updates_target_and_cr_frames() -> None:
    """The target echo block should place Z180 updates on target and CR channels."""
    block = health._target_cr_rotating_frame_echo_block(
        _experiment(), "Q0", "Q1", _schedule(20.0)
    )

    target_elements = block.get_sequences(copy=True)["Q1"].flattened_elements
    cr_elements = block.get_sequences(copy=True)["Q0-Q1"].flattened_elements
    target_z = [item for item in target_elements if isinstance(item, VirtualZ)]
    cr_z = [item for item in cr_elements if isinstance(item, VirtualZ)]

    assert [abs(item.theta) for item in target_z] == pytest.approx([np.pi, np.pi])
    assert [abs(item.theta) for item in cr_z] == pytest.approx([np.pi, np.pi])
    assert sum(isinstance(item, Rect) for item in cr_elements) == 4
    assert block.duration == pytest.approx(4 * 20.0 + 2 * 4.0)


def test_zx90_timing_accepts_metadata_exposed_by_properties() -> None:
    """Timing extraction should honor schedule metadata exposed as properties."""

    class _PropertySchedule:
        duration = 160.0

        @property
        def cr_duration(self) -> float:
            return 50.0

        @property
        def echo(self) -> bool:
            return True

    timing = health._extract_zx90_timing(cast(PulseSchedule, _PropertySchedule()))

    assert timing == health.ZX90Timing(
        cr_duration=50.0,
        echo=True,
        total_duration=160.0,
    )
    assert timing.cr_active_duration == pytest.approx(100.0)


def test_zx90_timing_rejects_boolean_schedule_duration() -> None:
    """Timing extraction should reject boolean schedule duration metadata."""
    schedule = SimpleNamespace(duration=True, cr_duration=0.5, echo=True)

    with pytest.raises(TypeError, match=r"duration.*real number"):
        health._extract_zx90_timing(cast(PulseSchedule, schedule))
