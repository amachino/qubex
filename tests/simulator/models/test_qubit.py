"""Tests for the lightweight qubit model."""

from __future__ import annotations

import pytest
from qxcore import units
from qxsimulator import Qubit


def test_qubit_defaults_to_zero_decoherence_rates() -> None:
    """Qubit should default to zero relaxation and dephasing rates."""
    qubit = Qubit(label="Q0", frequency=5.0)

    assert qubit.relaxation_rate == 0.0
    assert qubit.dephasing_rate == 0.0


def test_qubit_converts_t1_and_t2_to_physical_rates() -> None:
    """Qubit should convert T1 and T2 into physical decoherence rates."""
    qubit = Qubit(label="Q0", frequency=5.0, t1=100.0, t2=80.0)

    assert qubit.relaxation_rate == pytest.approx(0.01, rel=1e-12, abs=0.0)
    assert qubit.dephasing_rate == pytest.approx(0.0075, rel=1e-12, abs=0.0)


def test_qubit_normalizes_tunits_inputs_to_float_storage() -> None:
    """Qubit should normalize tunits frequency and coherence times to floats."""
    qubit = Qubit(
        label="Q0",
        frequency=5100 * units.MHz,
        t1=20 * units.us,
        t2=15 * units.us,
    )

    assert qubit.frequency == pytest.approx(5.1, rel=1e-12, abs=0.0)
    assert qubit.relaxation_rate == pytest.approx(1 / 20_000, rel=1e-12, abs=0.0)
    assert qubit.dephasing_rate == pytest.approx(
        1 / 15_000 - 1 / 40_000,
        rel=1e-12,
        abs=0.0,
    )
    assert isinstance(qubit.frequency, float)
    assert isinstance(qubit.relaxation_rate, float)
    assert isinstance(qubit.dephasing_rate, float)


def test_qubit_accepts_explicit_physical_rates() -> None:
    """Qubit should store explicit relaxation and dephasing rates unchanged."""
    qubit = Qubit(
        label="Q0",
        frequency=5.0,
        relaxation_rate=0.01,
        dephasing_rate=0.0075,
    )

    assert qubit.relaxation_rate == pytest.approx(0.01, rel=1e-12, abs=0.0)
    assert qubit.dephasing_rate == pytest.approx(0.0075, rel=1e-12, abs=0.0)


def test_qubit_normalizes_tunits_rates_to_inverse_ns() -> None:
    """Qubit should normalize tunits decay rates to inverse ns floats."""
    qubit = Qubit(
        label="Q0",
        frequency=5.0,
        relaxation_rate=1 * units.MHz,
        dephasing_rate=250 * units.kHz,
    )

    assert qubit.relaxation_rate == pytest.approx(0.001, rel=1e-12, abs=0.0)
    assert qubit.dephasing_rate == pytest.approx(0.00025, rel=1e-12, abs=0.0)


def test_qubit_rejects_mixed_time_and_rate_inputs() -> None:
    """Qubit should reject mixed coherence-time and rate parameterizations."""
    with pytest.raises(ValueError, match="either coherence times or rates"):
        Qubit(label="Q0", frequency=5.0, t1=100.0, relaxation_rate=0.01)


@pytest.mark.parametrize(
    ("t1", "t2", "parameter"),
    [(0.0, None, "t1"), (None, 0.0, "t2")],
)
def test_qubit_rejects_nonpositive_coherence_times(
    t1: float | None,
    t2: float | None,
    parameter: str,
) -> None:
    """Qubit should reject nonpositive coherence times."""
    with pytest.raises(ValueError, match=parameter):
        Qubit(label="Q0", frequency=5.0, t1=t1, t2=t2)


@pytest.mark.parametrize(
    ("relaxation_rate", "dephasing_rate", "parameter"),
    [(-0.01, None, "relaxation_rate"), (None, -0.01, "dephasing_rate")],
)
def test_qubit_rejects_negative_rates(
    relaxation_rate: float | None,
    dephasing_rate: float | None,
    parameter: str,
) -> None:
    """Qubit should reject negative decoherence rates."""
    with pytest.raises(ValueError, match=parameter):
        Qubit(
            label="Q0",
            frequency=5.0,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
        )


def test_qubit_rejects_t2_above_relaxation_limit() -> None:
    """Qubit should reject T2 values that imply negative pure dephasing."""
    with pytest.raises(ValueError, match="t2 must not exceed 2 × t1"):
        Qubit(label="Q0", frequency=5.0, t1=100.0, t2=201.0)
