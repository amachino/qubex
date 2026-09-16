"""Fast simulator and semantic pulse tests for the v11 specification."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from qubex.contrib.experiment._cr_dissipation_simulation import (
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
from qubex.contrib.experiment._cr_dissipation_types import IdleNoiseParameters


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
