"""Fast simulator and semantic pulse tests for the v11 specification."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from qubex.contrib.experiment._cr_dissipation.pulses import (
    PROTOCOLS,
    build_protocol_schedules,
    build_reference_schedules,
    resolve_zx90_descriptor,
    semantic_un_echoed_zx90,
    semantic_zx90,
)
from qubex.contrib.experiment._cr_dissipation.simulation import (
    X_TARGET,
    ZX,
    CrNoiseRates,
    F,
    G,
    SemanticSegment,
    compose_channel,
    leakage_aware_average_fidelity,
    propagate_fidelity_uncertainty,
    simultaneous_rotation_segments,
    tensor,
)
from qubex.contrib.experiment._cr_dissipation.types import IdleNoiseParameters
from qubex.experiment import Experiment
from qubex.pulse import CrossResonance, FlatTop, VirtualZ


def _pulse(duration: float = 20.0) -> FlatTop:
    """Build a lightweight calibrated waveform for schedule tests."""
    return FlatTop(duration=duration, amplitude=0.2, tau=4.0)


def _echoed_zx90() -> CrossResonance:
    """Build an echoed CR gate with a nonzero internal-pulse margin."""
    return CrossResonance(
        "Q0",
        "Q1",
        cr_amplitude=0.1,
        cr_duration=40.0,
        cr_ramptime=8.0,
        cancel_amplitude=0.03,
        echo=True,
        pi_pulse=_pulse(),
        pi_margin=4.0,
    )


def _schedule_experiment(echoed: CrossResonance) -> Experiment:
    """Build the pulse-service subset used by protocol schedule builders."""
    pulse = SimpleNamespace(
        zx90=lambda *_args, **_kwargs: echoed,
        get_pulse_for_state=lambda *_args: _pulse(),
        x90=lambda _target: _pulse(),
        x90m=lambda _target: _pulse(),
        y90=lambda _target: _pulse(),
        x180=lambda _target: _pulse(32.0),
        y180=lambda _target: _pulse(40.0),
        z180=lambda: VirtualZ(np.pi),
    )
    return cast(Experiment, SimpleNamespace(pulse=pulse))


def test_un_echoed_ab_gate_preserves_positive_calibrated_cr_lobes() -> None:
    """A/B use two same-sign calibrated lobes and duration-matched blank slots."""
    echoed = _echoed_zx90()
    descriptor = resolve_zx90_descriptor(_schedule_experiment(echoed), "Q0", "Q1", None)

    assert descriptor.full_un_echoed.duration == pytest.approx(echoed.duration)
    assert descriptor.echo_slot_duration_ns == pytest.approx(28.0)
    np.testing.assert_allclose(
        descriptor.full_un_echoed.values["Q0-Q1"][: echoed.cr_waveform.length],
        echoed.cr_waveform.values,
    )
    np.testing.assert_allclose(
        descriptor.full_un_echoed.values["Q1"][: echoed.cancel_waveform.length],
        echoed.cancel_waveform.values,
    )
    assert [operation.label for operation in semantic_un_echoed_zx90(descriptor)] == [
        "ZX45(+1)",
        "echo-slot-blank",
        "ZX45(+1)",
        "echo-slot-blank",
    ]
    assert [operation.label for operation in semantic_zx90(descriptor)] == [
        "ZX45(+1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "ZX45(-1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
    ]


def test_reference_protocols_preserve_actual_schedule_durations() -> None:
    """Every diagnostic reference remains duration-matched to its actual protocol."""
    echoed = _echoed_zx90()
    exp = _schedule_experiment(echoed)
    descriptor = resolve_zx90_descriptor(exp, "Q0", "Q1", None)
    counts = (0, 1, 2)
    actual = build_protocol_schedules(exp, "Q0", "Q1", descriptor, counts)
    reference = build_reference_schedules(
        exp,
        "Q0",
        "Q1",
        descriptor,
        counts,
        FlatTop(duration=40.0, amplitude=0.08, tau=8.0),
    )

    for protocol in PROTOCOLS:
        assert [schedule.duration for schedule in reference[protocol]] == pytest.approx(
            [schedule.duration for schedule in actual.actual[protocol]]
        )


def test_semantic_rotary_metadata_does_not_modify_actual_waveforms() -> None:
    """Semantic rotary metadata augments the model without changing measured pulses."""
    echoed = _echoed_zx90()
    echoed.rotary_integrated_angle_rad = 0.2
    echoed.rotary_phase_rad = 0.3
    descriptor = resolve_zx90_descriptor(_schedule_experiment(echoed), "Q0", "Q1", None)
    actual_before = descriptor.full_un_echoed.values

    positive = semantic_un_echoed_zx90(descriptor)[0]
    negative = semantic_zx90(descriptor)[4]

    np.testing.assert_allclose(negative.hamiltonian, -positive.hamiltonian)
    for label, waveform in actual_before.items():
        np.testing.assert_allclose(descriptor.full_un_echoed.values[label], waveform)
    assert not descriptor.warnings


def test_qutrit_embedding_stops_zx_but_not_unconditional_ix_after_control_leakage() -> (
    None
):
    """ZX annihilates control-f while unconditional target rotation remains."""
    leaked_control = np.outer(F, F.conj())
    target_ground = np.outer(G, G.conj())
    state = tensor(leaked_control, target_ground)

    np.testing.assert_allclose(ZX @ state, 0.0, atol=1e-14)
    assert np.linalg.norm(X_TARGET @ state) > 0.0


def test_simultaneous_pulses_do_not_stretch_the_shorter_rotation() -> None:
    """Piecewise layers stop the short Hamiltonian at its calibrated duration."""
    segments = simultaneous_rotation_segments(
        ((ZX, np.pi / 4.0, 20.0, "short"), (X_TARGET, np.pi, 50.0, "long"))
    )

    assert [segment.duration_ns for segment in segments] == [20.0, 30.0]
    assert segments[0].label == "short+long"
    assert segments[1].label == "long"


def test_leakage_aware_fidelity_is_one_for_identity_channel() -> None:
    """The generalized `(d Fe + s)/(d+1)` formula reduces to unity."""
    channel = np.eye(81, dtype=np.complex128)

    average, entanglement, survival = leakage_aware_average_fidelity(channel, channel)

    assert average == pytest.approx(1.0)
    assert entanglement == pytest.approx(1.0)
    assert survival == pytest.approx(1.0)


def test_signed_diagnostic_accepts_negative_coefficient_only_explicitly() -> None:
    """Negative pure dephasing is represented as a signed Liouvillian term."""
    idle = IdleNoiseParameters(float("inf"), float("inf"))
    operation = (
        SemanticSegment(10.0, np.zeros((9, 9), dtype=np.complex128), True, "CR"),
    )

    diagnostic = compose_channel(
        operation,
        control_idle=idle,
        target_idle=idle,
        cr_rates=CrNoiseRates(),
        include_leakage=False,
        signed_control_pure_dephasing=-1e-5,
    )

    assert np.all(np.isfinite(diagnostic))
    with pytest.raises(ValueError, match="nonnegative"):
        compose_channel(
            operation,
            control_idle=idle,
            target_idle=idle,
            cr_rates=CrNoiseRates(control_e_to_g=-1e-5),
            include_leakage=False,
        )


def test_unscented_covariance_psd_handling_and_positive_rates() -> None:
    """Log sigma points stay positive and material non-PSD input is rejected."""
    seen: list[float] = []

    def evaluator(rates: Mapping[str, float]) -> float:
        seen.append(rates["gamma"])
        return float(np.exp(-rates["gamma"]))

    error, ok = propagate_fidelity_uncertainty(
        {"gamma": 0.1},
        np.array([[0.0025]]),
        evaluator,
        positive_parameters=("gamma",),
    )

    assert ok
    assert error is not None
    assert error > 0.0
    assert min(seen) > 0.0
    error, ok = propagate_fidelity_uncertainty(
        {"a": 1.0, "b": 1.0},
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        lambda _: 1.0,
        positive_parameters=(),
    )
    assert error is None
    assert not ok
