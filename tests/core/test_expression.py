import numpy as np
import pytest

from qubex.core.expression import Expression


def test_expression_basic():
    expr = Expression("a + b")
    assert "a" in [s.name for s in expr.symbols]
    assert "b" in [s.name for s in expr.symbols]

    result = expr.resolve({"a": 1, "b": 2})
    assert result == 3


def test_expression_numpy():
    expr = Expression("x**2")
    x_vals = np.array([1, 2, 3])
    result = expr.resolve({"x": x_vals})
    np.testing.assert_array_equal(result, [1, 4, 9])


def test_expression_missing_symbol():
    expr = Expression("x + y")
    with pytest.raises(ValueError, match="Value for symbol 'y' not provided"):
        expr.resolve({"x": 1})


def test_expression_invalid_expression():
    with pytest.raises(ValueError, match="Failed to parse expression"):
        Expression("x + +")


def test_deterministic_symbol_order():
    # This was a requirement in the code review
    expr = Expression("c + b + a")
    symbol_names = [s.name for s in expr.symbols]
    # Check that symbols are sorted regardless of appearance order
    assert symbol_names == ["a", "b", "c"]
