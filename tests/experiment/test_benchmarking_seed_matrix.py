"""Seed matrix contracts for RB acquisition."""

from typing import Any

import numpy as np
import pytest

from qubex.experiment.services.benchmarking_service import (
    _resolve_rb_seeds as resolve_rb_seeds,
)


def test_seed_matrix_generates_one_seed_per_cell() -> None:
    """Generated seeds should cover every length and trial independently."""
    seeds = resolve_rb_seeds(None, n_lengths=4, n_trials=30)
    assert seeds.shape == (4, 30)
    assert len(np.unique(seeds)) == seeds.size


@pytest.mark.parametrize(
    "value", [[1, 2], [[1.5, 2.0]], [[-1, 2]], [[2**32, 2]], [[True, False]]]
)
def test_seed_matrix_rejects_invalid_inputs(value: Any) -> None:
    """Invalid seed matrices should fail before acquisition."""
    with pytest.raises((TypeError, ValueError)):
        resolve_rb_seeds(value, n_lengths=1, n_trials=2)


def test_seed_matrix_preserves_explicit_cells() -> None:
    """Explicit seeds should retain their length and trial correspondence."""
    expected = np.array([[11, 22], [33, 44]])
    actual = resolve_rb_seeds(expected, n_lengths=2, n_trials=2)
    np.testing.assert_array_equal(actual, expected)
