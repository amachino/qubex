"""Tests for the lightweight resonator model."""

from __future__ import annotations

import pytest
from qxcore import units
from qxsimulator import Resonator


def test_resonator_normalizes_none_rates_to_zero() -> None:
    """Resonator should normalize unspecified rates to zero."""
    resonator = Resonator(
        label="R0",
        dimension=5,
        frequency=7.0,
        relaxation_rate=None,
        dephasing_rate=None,
    )

    assert resonator.relaxation_rate == 0.0
    assert resonator.dephasing_rate == 0.0


def test_resonator_accepts_explicit_physical_rates() -> None:
    """Resonator should store explicit relaxation and dephasing rates unchanged."""
    resonator = Resonator(
        label="R0",
        dimension=5,
        frequency=7.0,
        relaxation_rate=0.01,
        dephasing_rate=0.002,
    )

    assert resonator.relaxation_rate == pytest.approx(0.01, rel=1e-12, abs=0.0)
    assert resonator.dephasing_rate == pytest.approx(0.002, rel=1e-12, abs=0.0)


def test_resonator_normalizes_tunits_frequency_and_rates() -> None:
    """Resonator should normalize tunits inputs to canonical floats."""
    resonator = Resonator(
        label="R0",
        dimension=5,
        frequency=7200 * units.MHz,
        relaxation_rate=2 * units.MHz,
        dephasing_rate=100 * units.kHz,
    )

    assert resonator.frequency == pytest.approx(7.2, rel=1e-12, abs=0.0)
    assert resonator.relaxation_rate == pytest.approx(0.002, rel=1e-12, abs=0.0)
    assert resonator.dephasing_rate == pytest.approx(0.0001, rel=1e-12, abs=0.0)
    assert isinstance(resonator.frequency, float)


def test_resonator_rejects_negative_rates() -> None:
    """Resonator should reject negative decoherence rates."""
    with pytest.raises(ValueError, match="relaxation_rate"):
        Resonator(
            label="R0",
            dimension=5,
            frequency=7.0,
            relaxation_rate=-0.01,
        )


def test_resonator_does_not_accept_coherence_times() -> None:
    """Resonator should keep its rate-only initialization API."""
    with pytest.raises(TypeError, match="t1"):
        Resonator(
            label="R0",
            dimension=5,
            frequency=7.0,
            t1=100.0,  # type: ignore[call-arg]
        )
