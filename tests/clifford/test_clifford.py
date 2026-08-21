"""Tests for Clifford transformation definitions."""

from qubex.clifford import Clifford, Pauli


def test_yi90_applies_positive_y90_to_first_qubit() -> None:
    """YI90 should tensor the positive Y90 map with identity on qubit two."""
    y90_map = Clifford.Y90().map
    expected = {
        first + second: Pauli(
            y90_map[first].coefficient,
            y90_map[first].operator + second,
        )
        for first in "IXYZ"
        for second in "IXYZ"
    }

    assert Clifford.YI90().map == expected
