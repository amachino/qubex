"""Schedule tests for readout-MIST acquisition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from qubex.contrib.experiment.readout_mist import (
    build_default_readout_cycle,
    build_mist_schedules,
    build_reference_cycle,
)
from qubex.experiment import Experiment
from qubex.pulse import FlatTop, PulseSchedule


class _Context:
    @staticmethod
    def resolve_ge_label(label: str) -> str:
        return label

    @staticmethod
    def resolve_ef_label(label: str) -> str:
        return f"{label}:ef"

    @staticmethod
    def resolve_read_label(label: str) -> str:
        return f"R{label}"


def _experiment() -> Experiment:
    pulse = SimpleNamespace(
        readout=lambda _target: FlatTop(duration=80.0, amplitude=0.2, tau=8.0),
        x180=lambda label: FlatTop(
            duration=32.0 if label.endswith(":ef") else 24.0,
            amplitude=0.1,
            tau=4.0,
        ),
    )
    return cast(Experiment, SimpleNamespace(ctx=_Context(), pulse=pulse))


def test_default_cycle_keeps_trailing_gap_in_every_repetition() -> None:
    """The last repetition should retain the same trailing gap as earlier cycles."""
    exp = _experiment()
    cycle = build_default_readout_cycle(exp, "Q0", cycle_gap_ns=20.0)

    schedules = build_mist_schedules(exp, "Q0", cycle, (0, 1, 2))

    assert cycle.duration == 100.0
    for protocol in ("initial_g", "initial_e", "initial_f"):
        base_duration = schedules.preparations[protocol].duration
        assert schedules.by_condition[protocol][0].duration == base_duration
        assert schedules.by_condition[protocol][1].duration == base_duration + 100.0
        assert schedules.by_condition[protocol][2].duration == base_duration + 200.0


def test_reference_cycle_is_a_duration_matched_blank_on_all_channels() -> None:
    """Reference construction should blank the complete candidate cycle."""
    with PulseSchedule(["RQ0", "DQ0"]) as candidate:
        candidate.add("RQ0", FlatTop(duration=80.0, amplitude=0.2, tau=8.0))
        candidate.add("DQ0", FlatTop(duration=100.0, amplitude=0.1, tau=8.0))
    candidate.set_frequencies({"RQ0": 8.1, "DQ0": 7.9})

    reference = build_reference_cycle(candidate)

    assert reference.duration == candidate.duration
    assert reference.labels == candidate.labels
    assert reference.get_frequencies() == candidate.get_frequencies()
    for values in reference.values.values():
        np.testing.assert_allclose(values, 0.0, rtol=0.0, atol=0.0)


def test_f_preparation_applies_ge_then_ef_pi_pulses() -> None:
    """The F protocol should prepare `f` with ordered GE and EF pi pulses."""
    exp = _experiment()
    cycle = build_default_readout_cycle(exp, "Q0", cycle_gap_ns=0.0)

    schedules = build_mist_schedules(exp, "Q0", cycle, (0, 1))

    preparation = schedules.preparations["initial_f"]
    assert preparation.duration == 56.0
    assert np.any(preparation.values["Q0"] != 0.0)
    assert np.any(preparation.values["Q0:ef"] != 0.0)
