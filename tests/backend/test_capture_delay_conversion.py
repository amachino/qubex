"""Tests for backend-owned capture-delay conversion rules."""

from __future__ import annotations

import pytest

from qubex.backend.quel1.capture_delay import (
    split_capture_delay_ns,
    validate_capture_delay_word,
)
from qubex.backend.quel3.capture_delay import normalize_capture_delay_ns


@pytest.mark.parametrize(
    ("delay_ns", "expected"),
    [(0, (0, 0)), (32, (0, 4)), (800.0, (6, 4)), (1024, (8, 0)), (1120, (8, 12))],
)
def test_quel1_split_capture_delay_ns(delay_ns, expected) -> None:
    """QuEL-1 total delays split into 128 ns coarse steps and a 32 ns word offset."""
    assert split_capture_delay_ns(delay_ns) == expected


@pytest.mark.parametrize("delay_ns", [8, 16, 24, 776.0, 800.5])
def test_quel1_split_capture_delay_ns_rejects_off_grid(delay_ns) -> None:
    """QuEL-1 delays outside the 32 ns grid raise without MUX context."""
    with pytest.raises(ValueError, match=r"^QuEL-1 capture delay must be a multiple"):
        split_capture_delay_ns(delay_ns)


@pytest.mark.parametrize("word", [1, -4, 4.5, True, "4"])
def test_quel1_validate_capture_delay_word_rejects_value_alone(word) -> None:
    """QuEL-1 word validation depends only on the value and omits MUX context."""
    with pytest.raises(
        ValueError, match=r"^QuEL-1 capture_delay_word must be .*multiple of 4 words"
    ):
        validate_capture_delay_word(word)


@pytest.mark.parametrize(
    ("delay_ns", "expected"), [(0, 0.0), (0.8, 0.8), (800.0, 800.0)]
)
def test_quel3_normalize_capture_delay_ns(delay_ns, expected) -> None:
    """QuEL-3 delays snap onto the 0.8 ns readout sampling grid."""
    assert normalize_capture_delay_ns(delay_ns) == pytest.approx(expected)


def test_quel3_normalize_capture_delay_ns_rejects_off_grid() -> None:
    """QuEL-3 delays outside the readout sampling grid raise without MUX context."""
    with pytest.raises(ValueError, match=r"^QuEL-3 capture delay must be a multiple"):
        normalize_capture_delay_ns(1.0)
