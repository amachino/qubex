"""Tests for the Expression helper."""

import numpy as np
import pytest

from qubex.core import Expression


def test_expression_basic():
    """Resolve a simple symbolic expression with scalars."""
    expr = Expression("a + b")
    assert "a" in [s.name for s in expr.symbols]
    assert "b" in [s.name for s in expr.symbols]

    result = expr.resolve({"a": 1, "b": 2})
    assert result == 3


def test_expression_numpy():
    """Resolve an expression with NumPy arrays."""
    expr = Expression("x**2")
    x_vals = np.array([1, 2, 3])
    result = expr.resolve({"x": x_vals})
    np.testing.assert_array_equal(result, [1, 4, 9])


def test_expression_missing_symbol():
    """Raise when a required symbol value is missing."""
    expr = Expression("x + y")
    with pytest.raises(ValueError, match="Value for symbol 'y' not provided"):
        expr.resolve({"x": 1})


def test_expression_invalid_expression():
    """Raise on invalid expression syntax."""
    with pytest.raises(ValueError, match="Failed to parse expression"):
        Expression("x + +")


def test_deterministic_symbol_order():
    """Ensure symbol order is deterministic for stable behavior."""
    # This was a requirement in the code review
    expr = Expression("c + b + a")
    symbol_names = [s.name for s in expr.symbols]
    # Check that symbols are sorted regardless of appearance order
    assert symbol_names == ["a", "b", "c"]
