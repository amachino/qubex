"""Tests for functional APIs in `qubex.contrib.experiment.cr_xt_decomposition`."""

from __future__ import annotations

from qubex.contrib import decompose_cr_crosstalk


def test_decompose_cr_crosstalk_is_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CR crosstalk decomposition helper is available."""
    assert callable(decompose_cr_crosstalk)
