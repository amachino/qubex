"""Compatibility tests for qubex.analysis.fitting wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from qxfitting import fitting as qxfitting_impl

from qubex.analysis import fitting as qubex_fitting


def test_fitting_wrapper_exports_qxfitting_symbols() -> None:
    """Given fitting wrapper, when reading core symbols, then they resolve to qxfitting implementations."""
    assert qubex_fitting.FitStatus is qxfitting_impl.FitStatus
    assert qubex_fitting.FitResult is qxfitting_impl.FitResult
    assert qubex_fitting.fit_linear is qxfitting_impl.fit_linear
    assert qubex_fitting.func_cos is qxfitting_impl.func_cos


def test_fit_linear_via_wrapper_returns_success() -> None:
    """Given linear data, when fit_linear runs via compatibility wrapper, then slope and status are correct."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x

    result = qubex_fitting.fit_linear(x, y, plot=False)

    assert result.status is qubex_fitting.FitStatus.SUCCESS
    assert result["a"] == pytest.approx(2.0, rel=1e-7, abs=1e-9)
