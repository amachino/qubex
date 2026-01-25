import numpy as np
import pytest

from qubex.core.variable import Variable


def test_variable_basic():
    var = Variable("a + b")
    assert "a" in [s.name for s in var.symbols]
    assert "b" in [s.name for s in var.symbols]

    result = var.resolve({"a": 1, "b": 2})
    assert result == 3


def test_variable_numpy():
    var = Variable("x**2")
    x_vals = np.array([1, 2, 3])
    result = var.resolve({"x": x_vals})
    np.testing.assert_array_equal(result, [1, 4, 9])


def test_variable_missing_symbol():
    var = Variable("x + y")
    with pytest.raises(ValueError, match="Value for symbol 'y' not provided"):
        var.resolve({"x": 1})


def test_variable_invalid_expression():
    with pytest.raises(ValueError, match="Failed to parse expression"):
        Variable("x + +")


def test_deterministic_symbol_order():
    # This was a requirement in the code review
    expr = "c + b + a"
    var = Variable(expr)
    symbol_names = [s.name for s in var.symbols]
    # Check that symbols are sorted regardless of appearance order
    assert symbol_names == ["a", "b", "c"]
