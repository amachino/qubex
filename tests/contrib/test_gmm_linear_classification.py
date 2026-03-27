"""Tests for experimental GMM-derived linear DSP classification helpers."""

from __future__ import annotations

import pytest

from qubex.contrib import (
    build_gmm_linear_line_param,
    build_gmm_linear_line_params,
)


def _evaluate_line(line: tuple[float, float, float], point: complex) -> float:
    a, b, c = line
    return a * point.real + b * point.imag + c


def test_build_gmm_linear_line_param_uses_normalized_perpendicular_bisector() -> None:
    """Given g/e centers, helper should return the midpoint bisector with g on the positive side."""
    line = build_gmm_linear_line_param({0: 1.0 + 1.0j, 1: 3.0 + 1.0j})

    assert _evaluate_line(line, 2.0 + 1.0j) == pytest.approx(0.0)
    assert _evaluate_line(line, 1.0 + 1.0j) > 0.0
    assert _evaluate_line(line, 3.0 + 1.0j) < 0.0
    assert line[0] ** 2 + line[1] ** 2 == pytest.approx(1.0)


def test_build_gmm_linear_line_params_returns_per_target_mapping() -> None:
    """Given per-target centers, helper should build one separator for each target."""
    line_params = build_gmm_linear_line_params(
        {
            "RQ00": {0: 0.0 + 0.0j, 1: 2.0 + 0.0j},
            "RQ01": {0: 1.0 + 1.0j, 1: 1.0 + 3.0j},
        }
    )

    assert set(line_params) == {"RQ00", "RQ01"}
    assert _evaluate_line(line_params["RQ00"], 0.0 + 0.0j) > 0.0
    assert _evaluate_line(line_params["RQ00"], 2.0 + 0.0j) < 0.0
    assert _evaluate_line(line_params["RQ01"], 1.0 + 1.0j) > 0.0
    assert _evaluate_line(line_params["RQ01"], 1.0 + 3.0j) < 0.0


def test_build_gmm_linear_line_param_rejects_identical_centers() -> None:
    """Given identical centers, helper should reject the degenerate separator."""
    with pytest.raises(ValueError, match="must not be identical"):
        build_gmm_linear_line_param({0: 1.0 + 0.0j, 1: 1.0 + 0.0j})
