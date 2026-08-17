"""Tests for the lightweight transmon model."""

from __future__ import annotations

import pytest
from qxcore import units
from qxsimulator import Transmon


def test_transmon_defaults_to_zero_decoherence_rates() -> None:
    """Transmon should default to zero relaxation and dephasing rates."""
    transmon = Transmon(label="Q0", dimension=3, frequency=5.0)

    assert transmon.relaxation_rate == 0.0
    assert transmon.dephasing_rate == 0.0


def test_transmon_converts_t1_and_t2_to_physical_rates() -> None:
    """Transmon should convert T1 and T2 into physical decoherence rates."""
    transmon = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        t1=100.0,
        t2=80.0,
    )

    assert transmon.relaxation_rate == pytest.approx(0.01, rel=1e-12, abs=0.0)
    assert transmon.dephasing_rate == pytest.approx(0.0075, rel=1e-12, abs=0.0)


def test_transmon_normalizes_tunits_inputs_to_float_storage() -> None:
    """Transmon should normalize tunits model parameters to canonical floats."""
    transmon = Transmon(
        label="Q0",
        dimension=4,
        frequency=5100 * units.MHz,
        anharmonicity=-250 * units.MHz,
        t1=20 * units.us,
        t2=15 * units.us,
    )

    assert transmon.frequency == pytest.approx(5.1, rel=1e-12, abs=0.0)
    assert transmon.anharmonicity == pytest.approx(-0.25, rel=1e-12, abs=0.0)
    assert transmon.relaxation_rate == pytest.approx(
        1 / 20_000,
        rel=1e-12,
        abs=0.0,
    )
    assert transmon.dephasing_rate == pytest.approx(
        1 / 15_000 - 1 / 40_000,
        rel=1e-12,
        abs=0.0,
    )
    assert isinstance(transmon.frequency, float)
    assert isinstance(transmon.anharmonicity, float)


def test_transmon_derives_default_anharmonicity_after_unit_normalization() -> None:
    """Transmon should derive its default anharmonicity in canonical GHz."""
    transmon = Transmon(
        label="Q0",
        dimension=4,
        frequency=5 * units.GHz,
    )

    assert transmon.frequency == pytest.approx(5.0, rel=1e-12, abs=0.0)
    assert transmon.anharmonicity == pytest.approx(-0.25, rel=1e-12, abs=0.0)


def test_cosine_transmon_uses_default_charge_cutoff() -> None:
    """Cosine transmon should default its charge-basis cutoff to 25."""
    transmon = Transmon(
        label="Q0",
        dimension=4,
        frequency=5.0,
        anharmonicity=-0.25,
        model="cosine",
    )

    assert transmon.model == "cosine"
    assert transmon.charge_cutoff == 25
    assert transmon.offset_charge == 0.0


def test_transmon_accepts_explicit_physical_rates() -> None:
    """Transmon should store explicit relaxation and dephasing rates unchanged."""
    transmon = Transmon(
        label="Q0",
        dimension=3,
        frequency=5.0,
        relaxation_rate=0.01,
        dephasing_rate=0.0075,
    )

    assert transmon.relaxation_rate == pytest.approx(0.01, rel=1e-12, abs=0.0)
    assert transmon.dephasing_rate == pytest.approx(0.0075, rel=1e-12, abs=0.0)


def test_transmon_rejects_mixed_time_and_rate_inputs() -> None:
    """Transmon should reject mixed coherence-time and rate parameterizations."""
    with pytest.raises(ValueError, match="either coherence times or rates"):
        Transmon(
            label="Q0",
            dimension=3,
            frequency=5.0,
            t1=100.0,
            relaxation_rate=0.01,
        )
