"""Compatibility tests for qubex.analysis.fitting wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.analysis import fitting as qubex_fitting


def test_qubex_fitting_exposes_legacy_symbols() -> None:
    """Given qubex fitting module, when reading core symbols, then legacy APIs are available."""
    assert hasattr(qubex_fitting, "FitStatus")
    assert hasattr(qubex_fitting, "FitResult")
    assert hasattr(qubex_fitting, "fit_linear")
    assert hasattr(qubex_fitting, "func_cos")


def test_fit_linear_via_wrapper_returns_success() -> None:
    """Given linear data, when fit_linear runs via qubex fitting, then slope and status are correct."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x

    result = qubex_fitting.fit_linear(x, y, plot=False)

    assert result.status is qubex_fitting.FitStatus.SUCCESS
    assert result["a"] == pytest.approx(2.0, rel=1e-7, abs=1e-9)


def test_qxfitting_is_placeholder_for_now() -> None:
    """Given qxfitting module, when checking public symbols, then legacy fit entry points are absent."""
    from qxfitting import fitting as qxfitting_impl

    assert not hasattr(qxfitting_impl, "fit_linear")
    assert not hasattr(qxfitting_impl, "func_cos")
