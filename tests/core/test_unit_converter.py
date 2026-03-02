"""Tests for quantity normalization helpers."""

from __future__ import annotations

import pytest

from qubex.core import Frequency, Time, units
from qubex.core.unit_converter import (
    normalize_frequencies_to_ghz,
    normalize_frequency_to_ghz,
    normalize_quantity,
    normalize_quantity_mapping,
    normalize_time_to_ns,
)


def test_normalize_quantity_converts_time_with_configurable_scale() -> None:
    """Given Time value, when normalize_quantity is used, then it converts via base-unit scaling."""
    result = normalize_quantity(
        Time(2, "us"),
        quantity_type=Time,
        scale_from_base=1e9,
    )

    assert result == pytest.approx(2000.0)


def test_normalize_quantity_preserves_plain_float_value() -> None:
    """Given plain float, when normalize_quantity is used, then it returns float value unchanged."""
    result = normalize_quantity(
        128.5,
        quantity_type=Time,
        scale_from_base=1e9,
    )

    assert result == pytest.approx(128.5)


def test_normalize_time_to_ns_converts_tunits_time() -> None:
    """Given tunits time value, when normalize_time_to_ns is used, then ns float is returned."""
    result = normalize_time_to_ns(2 * units.us)

    assert result == pytest.approx(2000.0)


def test_normalize_frequency_to_ghz_converts_tunits_frequency() -> None:
    """Given tunits frequency value, when normalize_frequency_to_ghz is used, then GHz float is returned."""
    result = normalize_frequency_to_ghz(5100 * units.MHz)

    assert result == pytest.approx(5.1)


def test_normalize_quantity_mapping_converts_each_value() -> None:
    """Given quantity mapping, when normalize_quantity_mapping is used, then all entries are normalized."""
    result = normalize_quantity_mapping(
        {
            "Q00": 5100 * units.MHz,
            "Q01": 5.2,
        },
        quantity_type=Frequency,
        scale_from_base=1e-9,
    )

    assert result is not None
    assert result["Q00"] == pytest.approx(5.1)
    assert result["Q01"] == pytest.approx(5.2)


def test_normalize_frequencies_to_ghz_converts_frequency_mapping() -> None:
    """Given frequency overrides, when normalize_frequencies_to_ghz is used, then GHz float mapping is returned."""
    result = normalize_frequencies_to_ghz(
        {
            "Q00": 5100 * units.MHz,
            "Q01": 5.2,
        }
    )

    assert result == {
        "Q00": pytest.approx(5.1),
        "Q01": pytest.approx(5.2),
    }
