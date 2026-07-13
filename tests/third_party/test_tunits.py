"""Tests for third-party `tunits` behavior used by Qubex."""

from __future__ import annotations

import tunits


def test_tunits_units():
    """tunits.units should return expected types."""
    time = 2 * tunits.units.ns
    frequency = 100 * tunits.units.MHz
    product = time * frequency

    # NOTE: tunits.units.*** is not typed as specific quantity type
    assert not isinstance(time, tunits.Time)
    assert not isinstance(frequency, tunits.Frequency)

    assert isinstance(time, tunits.Value)
    assert time.value == 2
    assert time.value_in_base_units() == 2e-9
    assert time.units == "ns"
    assert time.is_dimensionless is False

    assert isinstance(frequency, tunits.Value)
    assert frequency.value == 100
    assert frequency.value_in_base_units() == 1e8
    assert frequency.units == "MHz"
    assert frequency.is_dimensionless is False

    assert isinstance(product, tunits.Value)
    assert product.value == 200
    assert product.value_in_base_units() == 0.2
    assert product.units == "MHz*ns"
    assert product.is_dimensionless is True


def test_tunits_units_with_dimension():
    """tunits.units_with_dimension should return expected types."""
    time = 2 * tunits.units_with_dimension.ns
    frequency = 100 * tunits.units_with_dimension.MHz
    product = time * frequency

    assert isinstance(time, tunits.Time)
    assert isinstance(frequency, tunits.Frequency)

    assert isinstance(time, tunits.Value)
    assert time.value == 2
    assert time.value_in_base_units() == 2e-9
    assert time.units == "ns"
    assert time.is_dimensionless is False

    assert isinstance(frequency, tunits.Value)
    assert frequency.value == 100
    assert frequency.value_in_base_units() == 1e8
    assert frequency.units == "MHz"
    assert frequency.is_dimensionless is False

    assert isinstance(product, tunits.Value)
    assert product.value == 200
    assert product.value_in_base_units() == 0.2
    assert product.units == "MHz*ns"
    assert product.is_dimensionless is True
