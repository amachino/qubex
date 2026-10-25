"""Numerical helper tests for readout-MIST fitting."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.contrib.experiment.readout_mist import (
    assess_goodness_of_fit,
    fit_full_gls,
)


def test_converged_central_fit_survives_unresolved_local_covariance() -> None:
    """A rank-deficient fit should retain covariance and use rank for DoF."""
    result = fit_full_gls(
        lambda parameters: np.array([parameters[0] - 0.25, parameters[0] - 0.25]),
        (np.array([0.0, 0.0]),),
        parameter_count=2,
    )

    assert result.success
    assert result.parameters[0] == pytest.approx(0.25, rel=0.0, abs=1e-8)
    assert result.jacobian_rank == 1
    assert result.jacobian_nullity == 1
    assert result.null_space.shape == (1, 2)
    assert np.all(np.isfinite(result.covariance))
    assert "not locally identifiable" in result.message
    assert result.goodness_of_fit.degrees_of_freedom == 1
    assert np.isfinite(result.goodness_of_fit.reduced_chi_squared)


def test_goodness_of_fit_reports_objective_residual_metrics_without_judgment() -> None:
    """Fit diagnostics should report chi-squared and whitened-residual RMS."""
    goodness = assess_goodness_of_fit(
        np.array([1.0, -2.0, 2.0, 0.0]),
        n_parameters=1,
    )

    assert goodness.degrees_of_freedom == 3
    assert goodness.chi_squared == pytest.approx(9.0)
    assert goodness.reduced_chi_squared == pytest.approx(3.0)
    assert goodness.weighted_residual_rms == pytest.approx(np.sqrt(3.0))
    assert goodness.maximum_absolute_whitened_residual == pytest.approx(2.0)


def test_local_covariance_is_not_scaled_by_fit_residual_size() -> None:
    """Local GLS uncertainty should remain separate from fit-quality metrics."""
    result = fit_full_gls(
        lambda parameters: np.array(
            [parameters[0] - 10.0, parameters[0] + 10.0, parameters[0]]
        ),
        (np.array([0.0]),),
        parameter_count=1,
    )

    assert result.goodness_of_fit.reduced_chi_squared > 1.0
    assert result.covariance[0, 0] == pytest.approx(1.0 / 3.0)
