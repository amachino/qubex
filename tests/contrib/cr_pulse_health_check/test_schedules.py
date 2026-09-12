"""Tests for CR-pulse health-check schedule construction."""

# ruff: noqa: SLF001

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import qubex.contrib.experiment.cr_pulse_health_check as health
from qubex.experiment import Experiment
from qubex.pulse import Blank, PulseSchedule, Rect, VirtualZ


class _DummyPulseService:
    """Provide deterministic pulses for schedule construction."""

    def x180(self, _target: str) -> Blank:
        """Return a four-nanosecond X180 stand-in."""
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


def test_control_t2_echo_block_uses_documented_x180_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control echo block should apply XI, XI+IX, and XI between four gates."""
    exp = _experiment()
    targets: list[str] = []
    original = exp.pulse.x180

    def record_x180(target: str) -> Blank:
        targets.append(target)
        return cast(Blank, original(target))

    monkeypatch.setattr(exp.pulse, "x180", record_x180)

    block = health._control_t2_echo_block(exp, "Q0", "Q1", _schedule(20.0))

    assert targets == ["Q0", "Q0", "Q1", "Q0"]
    assert block.duration == pytest.approx(4 * 20.0 + 3 * 4.0)


def test_target_t2rho_block_updates_target_and_cr_frames() -> None:
    """The target echo block should place Z180 updates on target and CR channels."""
    block = health._target_t2rho_echo_block(_experiment(), "Q0", "Q1", _schedule(20.0))

    target_elements = block.get_sequences(copy=True)["Q1"].flattened_elements
    cr_elements = block.get_sequences(copy=True)["Q0-Q1"].flattened_elements
    target_z = [item for item in target_elements if isinstance(item, VirtualZ)]
    cr_z = [item for item in cr_elements if isinstance(item, VirtualZ)]

    assert [abs(item.theta) for item in target_z] == pytest.approx([np.pi])
    assert [abs(item.theta) for item in cr_z] == pytest.approx([np.pi])
    assert sum(isinstance(item, Rect) for item in cr_elements) == 2
    assert block.duration == pytest.approx(40.0)


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
        cr_lobe_duration=50.0,
        echo=True,
        total_duration=160.0,
    )
    assert timing.cr_active_duration == pytest.approx(100.0)


def test_zx90_timing_rejects_boolean_schedule_duration() -> None:
    """Timing extraction should reject boolean schedule duration metadata."""
    schedule = SimpleNamespace(duration=True, cr_duration=0.5, echo=True)

    with pytest.raises(TypeError, match=r"duration.*real number"):
        health._extract_zx90_timing(cast(PulseSchedule, schedule))
