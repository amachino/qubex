"""Test CR dissipation schedules and semantic two-qutrit simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.linalg import expm

import qubex.contrib.experiment._cr_dissipation.simulation as simulation_module
from qubex.contrib.experiment._cr_dissipation.pulses import (
    PROTOCOL_D,
    PROTOCOLS,
    build_protocol_schedules,
    build_reference_schedules,
    resolve_zx90_descriptor,
    semantic_protocol_blocks,
    semantic_reference_blocks,
    semantic_un_echoed_zx90,
    semantic_zx90,
)
from qubex.contrib.experiment._cr_dissipation.simulation import (
    X_CONTROL,
    X_TARGET,
    Y_TARGET,
    Z_TARGET,
    CrNoiseRates,
    SemanticSegment,
    SemanticUnitary,
    compose_channel,
    propagate_fidelity_uncertainty,
    simulate_repeated_observable,
    state_density,
)
from qubex.contrib.experiment._cr_dissipation.types import IdleNoiseParameters
from qubex.experiment import Experiment
from qubex.pulse import CrossResonance, FlatTop, VirtualZ


def _pulse(duration: float = 20.0) -> FlatTop:
    """Build a lightweight calibrated waveform for schedule tests."""
    return FlatTop(duration=duration, amplitude=0.2, tau=4.0)


def test_segment_superoperator_cache_reuses_matrix_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical segments reuse their expensive matrix exponential."""
    segment = SemanticSegment(
        20.0,
        np.zeros((9, 9), dtype=np.complex128),
        True,
        "CR",
    )
    idle = IdleNoiseParameters(float("inf"), float("inf"))
    cache: simulation_module.SuperoperatorCache = {}
    original_expm = simulation_module.expm
    calls = 0

    def counted_expm(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nonlocal calls
        calls += 1
        return original_expm(matrix)

    monkeypatch.setattr(simulation_module, "expm", counted_expm)
    for _ in range(2):
        compose_channel(
            (segment, segment),
            control_idle=idle,
            target_idle=idle,
            cr_rates=CrNoiseRates(),
            include_leakage=True,
            cache=cache,
        )

    assert calls == 1


def test_repeated_observable_matches_independent_channel_powers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared binary powers preserve arbitrary repeated-count predictions."""
    idle = IdleNoiseParameters(40_000.0, 50_000.0)
    operations = (
        SemanticSegment(
            20.0,
            np.zeros((9, 9), dtype=np.complex128),
            True,
            "CR",
        ),
    )
    rates = CrNoiseRates(control_e_to_g=2e-5, target_t1rho=3e-5)
    initial = state_density("+x", "+x")
    counts = np.array([5, 0, 2, 5], dtype=np.int64)
    channel = compose_channel(
        operations,
        control_idle=idle,
        target_idle=idle,
        cr_rates=rates,
        include_leakage=True,
    )
    expected = np.array(
        [
            np.real(
                np.trace(
                    simulation_module.apply_channel(
                        np.linalg.matrix_power(channel, int(count)),
                        initial,
                    )
                    @ X_CONTROL
                )
            )
            for count in counts
        ]
    )
    monkeypatch.setattr(
        simulation_module.np.linalg,
        "matrix_power",
        lambda *_args, **_kwargs: pytest.fail(
            "repeated-observable simulation rebuilt an independent matrix power"
        ),
    )

    actual = simulate_repeated_observable(
        operations,
        counts,
        initial,
        X_CONTROL,
        control_idle=idle,
        target_idle=idle,
        cr_rates=rates,
        include_leakage=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


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

    def zx90(*_args: object, **kwargs: object) -> CrossResonance:
        if kwargs.get("rotary_amplitude") != 0.0:
            return echoed
        return CrossResonance(
            "Q0",
            "Q1",
            cr_amplitude=echoed.cr_amplitude,
            cr_duration=echoed.cr_duration,
            cr_ramptime=echoed.cr_ramptime,
            cr_phase=echoed.cr_phase,
            cr_beta=echoed.cr_beta,
            cancel_amplitude=0.02,
            cancel_phase=echoed.cancel_phase,
            cancel_beta=echoed.cancel_beta,
            echo=True,
            pi_pulse=echoed.pi_pulse,
            pi_margin=4.0,
        )

    pulse = SimpleNamespace(
        zx90=zx90,
        get_pulse_for_state=lambda *_args: _pulse(),
        x90=lambda _target: _pulse(),
        x90m=lambda _target: _pulse(),
        y90=lambda _target: _pulse(),
        x180=lambda _target: _pulse(32.0),
        y180=lambda _target: _pulse(40.0),
        z180=lambda: VirtualZ(np.pi),
    )
    return cast(Experiment, SimpleNamespace(pulse=pulse))


def test_un_echoed_ab_gate_removes_rotary_but_preserves_cr_and_cancel() -> None:
    """A/B remove rotary while preserving CR, cancel, and matched blank slots."""
    echoed = _echoed_zx90()
    descriptor = resolve_zx90_descriptor(_schedule_experiment(echoed), "Q0", "Q1", None)

    assert descriptor.full_un_echoed.duration == pytest.approx(echoed.duration)
    assert descriptor.echo_slot_duration_ns == pytest.approx(28.0)
    np.testing.assert_allclose(
        descriptor.full_un_echoed.values["Q0-Q1"][: echoed.cr_waveform.length],
        echoed.cr_waveform.values,
    )
    assert not np.allclose(
        descriptor.full_un_echoed.values["Q1"][: echoed.cancel_waveform.length],
        echoed.cancel_waveform.values,
    )
    expected_cancel = FlatTop(
        duration=echoed.cr_duration,
        amplitude=0.02,
        tau=echoed.cr_ramptime,
        phase=echoed.cancel_phase,
        beta=echoed.cancel_beta,
    )
    np.testing.assert_allclose(
        descriptor.full_un_echoed.values["Q1"][: expected_cancel.length],
        expected_cancel.values,
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


def test_protocol_d_uses_ground_state_z_readout_without_analysis_rotation() -> None:
    """Protocol D prepares target ground and uses direct GEF Z readout."""
    echoed = _echoed_zx90()
    exp = _schedule_experiment(echoed)
    descriptor = resolve_zx90_descriptor(exp, "Q0", "Q1", None)
    schedules = build_protocol_schedules(exp, "Q0", "Q1", descriptor, (0, 1))

    assert schedules.actual[PROTOCOL_D][0].duration == pytest.approx(
        schedules.base[PROTOCOL_D][0].duration
    )
    assert schedules.timing[PROTOCOL_D]["operation_order"] == (
        "ZX90",
        "IZ180",
        "ZX90",
        "IY180",
        "ZX90",
        "IZ180",
        "ZX90",
        "IY180",
    )
    assert [operation.label for operation in schedules.semantic_blocks[PROTOCOL_D]] == [
        "ZX45(+1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "ZX45(-1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "IZ180(target+CR-frame)",
        "ZX45(+1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "ZX45(-1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "IY180",
        "ZX45(+1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "ZX45(-1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "IZ180(target+CR-frame)",
        "ZX45(+1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "ZX45(-1)",
        "echo-margin",
        "XI180(internal)",
        "echo-margin",
        "IY180",
    ]


def test_protocol_d_ideal_block_refocuses_z_even_with_zx_angle_error() -> None:
    """The swapped D block returns ground-state Z for a common ZX angle error."""
    descriptor = resolve_zx90_descriptor(
        _schedule_experiment(_echoed_zx90()), "Q0", "Q1", None
    )
    block = semantic_protocol_blocks(
        descriptor,
        external_durations_ns={
            "xi180": 32.0,
            "yi180": 40.0,
            "ix180": 32.0,
            "iy180": 40.0,
        },
    )[PROTOCOL_D]
    detuned = tuple(
        replace(operation, hamiltonian=1.13 * operation.hamiltonian)
        if isinstance(operation, SemanticSegment) and operation.label.startswith("ZX45")
        else operation
        for operation in block
    )
    initial = state_density("g", "g")

    final = initial
    for operation in detuned:
        unitary = (
            operation.unitary
            if isinstance(operation, SemanticUnitary)
            else expm(-1.0j * operation.hamiltonian * operation.duration_ns)
        )
        final = unitary @ final @ unitary.conj().T
    for observable, expected in (
        (Z_TARGET, 1.0),
        (X_TARGET, 0.0),
        (Y_TARGET, 0.0),
    ):
        assert np.real(np.trace(initial @ observable)) == pytest.approx(expected)
        assert np.real(np.trace(final @ observable)) == pytest.approx(
            expected, abs=1e-12
        )


def test_cd_reference_semantics_match_blank_and_ix45_replacements() -> None:
    """C uses blank lobes and D uses IX45 lobes in reference predictions."""
    descriptor = resolve_zx90_descriptor(
        _schedule_experiment(_echoed_zx90()), "Q0", "Q1", None
    )
    durations = {
        "xi180": 32.0,
        "yi180": 40.0,
        "ix180": 32.0,
        "iy180": 40.0,
    }
    blocks = semantic_reference_blocks(
        descriptor,
        external_durations_ns=durations,
    )
    actual = semantic_protocol_blocks(descriptor, external_durations_ns=durations)

    c_labels = [operation.label for operation in blocks["control_cr_transverse_echo"]]
    d_labels = [operation.label for operation in blocks[PROTOCOL_D]]
    assert c_labels.count("blank-reference") == 8
    assert "IX45(reference)" not in c_labels
    assert d_labels.count("IX45(reference)") == 8
    assert all(
        not operation.cr_active
        for block in blocks.values()
        for operation in block
        if isinstance(operation, SemanticSegment)
    )
    for protocol in ("control_cr_transverse_echo", PROTOCOL_D):
        reference_duration = sum(
            operation.duration_ns
            for operation in blocks[protocol]
            if isinstance(operation, SemanticSegment)
        )
        actual_duration = sum(
            operation.duration_ns
            for operation in actual[protocol]
            if isinstance(operation, SemanticSegment)
        )
        assert reference_duration == pytest.approx(actual_duration)


def test_protocol_d_dissipative_forward_model_uses_target_z() -> None:
    """D target rotating-frame dephasing reduces the simulated Z contrast."""
    descriptor = resolve_zx90_descriptor(
        _schedule_experiment(_echoed_zx90()), "Q0", "Q1", None
    )
    block = semantic_protocol_blocks(
        descriptor,
        external_durations_ns={
            "xi180": 32.0,
            "yi180": 40.0,
            "ix180": 32.0,
            "iy180": 40.0,
        },
    )[PROTOCOL_D]
    idle = IdleNoiseParameters(float("inf"), float("inf"))

    values = simulate_repeated_observable(
        block,
        (0, 1, 2),
        state_density("g", "g"),
        Z_TARGET,
        control_idle=idle,
        target_idle=idle,
        cr_rates=CrNoiseRates(target_rotating_frame_pure_dephasing=1e-3),
        include_leakage=True,
    )

    assert values[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(values))
    assert values[-1] < values[1] < values[0]


def test_semantic_rotary_is_used_for_echoed_protocols_but_not_ab() -> None:
    """Rotary metadata affects echoed C/D semantics but is excluded from A/B."""
    echoed = _echoed_zx90()
    echoed_with_metadata = cast(Any, echoed)
    echoed_with_metadata.rotary_integrated_angle_rad = 0.2
    echoed_with_metadata.rotary_phase_rad = 0.3
    descriptor = resolve_zx90_descriptor(_schedule_experiment(echoed), "Q0", "Q1", None)
    actual_before = descriptor.full_un_echoed.values

    ab_positive = semantic_un_echoed_zx90(descriptor)[0]
    positive = semantic_zx90(descriptor)[0]
    negative = semantic_zx90(descriptor)[4]

    assert isinstance(ab_positive, SemanticSegment)
    assert isinstance(positive, SemanticSegment)
    assert isinstance(negative, SemanticSegment)
    np.testing.assert_allclose(negative.hamiltonian, -positive.hamiltonian)
    assert not np.allclose(ab_positive.hamiltonian, positive.hamiltonian)
    for label, waveform in actual_before.items():
        np.testing.assert_allclose(descriptor.full_un_echoed.values[label], waveform)


def test_control_e_to_f_leakage_adds_half_rate_ge_coherence_decay() -> None:
    """Control e-to-f leakage damps ge coherence at half the jump rate."""
    rate = 3e-4
    duration = 200.0
    idle = IdleNoiseParameters(float("inf"), float("inf"))
    operation = (
        SemanticSegment(
            duration,
            np.zeros((9, 9), dtype=np.complex128),
            True,
            "CR",
        ),
    )

    value = simulate_repeated_observable(
        operation,
        (1,),
        state_density("+x", "g"),
        X_CONTROL,
        control_idle=idle,
        target_idle=idle,
        cr_rates=CrNoiseRates(control_e_to_f=rate),
        include_leakage=True,
    )[0]
    coherence_only = simulate_repeated_observable(
        operation,
        (1,),
        state_density("+x", "g"),
        X_CONTROL,
        control_idle=idle,
        target_idle=idle,
        cr_rates=CrNoiseRates(control_e_to_f=rate),
        include_leakage=False,
    )[0]

    assert value == pytest.approx(np.exp(-0.5 * rate * duration), rel=1e-12)
    assert coherence_only == pytest.approx(1.0, abs=1e-12)


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
