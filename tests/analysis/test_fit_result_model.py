"""Tests for the FitResult model."""

from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go
import pytest

from qubex.analysis.fit_result import FitResult, FitStatus


def test_fit_result_should_preserve_payload_and_metadata() -> None:
    """FitResult should preserve payload, status, and metadata."""
    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={"value": 1.0},
    )

    assert result.status is FitStatus.SUCCESS
    assert result.message == "ok"
    assert result["value"] == 1.0
    assert result["status"] == FitStatus.SUCCESS.value
    assert datetime.fromisoformat(result.created_at).tzinfo == timezone.utc
    assert result.figure is None
    assert result.figures is None


def test_fit_result_should_not_derive_figure_fields_from_legacy_payload_keys() -> None:
    """FitResult should not derive figure attributes from legacy payload keys."""
    figure = go.Figure()
    figure_3d = go.Figure()

    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={
            "fig": figure,
            "fig3d": figure_3d,
        },
    )

    assert result.figure is None
    assert result.figures is None


def test_fit_result_should_warn_on_legacy_figure_key_access() -> None:
    """FitResult should warn when legacy figure payload keys are accessed."""
    figure = go.Figure()
    figure_3d = go.Figure()
    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={
            "fig": figure,
            "fig3d": figure_3d,
        },
    )

    with pytest.warns(DeprecationWarning, match="figure` attribute"):
        assert result["fig"] is figure

    with pytest.warns(DeprecationWarning, match="figures` attribute"):
        assert result.data["fig3d"] is figure_3d


def test_fit_result_should_prefer_explicit_figure_fields() -> None:
    """FitResult should prefer explicit figure fields over legacy payload keys."""
    explicit_figure = go.Figure()
    explicit_figures = {"fig3d": go.Figure()}
    legacy_figure = go.Figure()
    legacy_figure_3d = go.Figure()

    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={
            "fig": legacy_figure,
            "fig3d": legacy_figure_3d,
        },
        figure=explicit_figure,
        figures=explicit_figures,
    )

    assert result.figure is explicit_figure
    assert result.figures == explicit_figures
    with pytest.warns(DeprecationWarning, match="figure` attribute"):
        assert result["fig"] is legacy_figure
    with pytest.warns(DeprecationWarning, match="figures` attribute"):
        assert result["fig3d"] is legacy_figure_3d


def test_fit_result_should_return_primary_and_named_figures_via_get_figure() -> None:
    """FitResult should return primary and named figures via get_figure."""
    figure = go.Figure()
    figures = {"fig3d": go.Figure(), "overview": go.Figure()}
    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={"value": 1.0},
        figure=figure,
        figures=figures,
    )

    assert result.get_figure() is figure
    assert result.get_figure("fig3d") is figures["fig3d"]


def test_fit_result_should_raise_when_requested_figure_is_missing() -> None:
    """FitResult should raise when the requested figure is missing."""
    result = FitResult(
        status=FitStatus.SUCCESS,
        message="ok",
        data={"value": 1.0},
    )

    with pytest.raises(ValueError, match="primary figure"):
        result.get_figure()

    with pytest.raises(KeyError, match="fig3d"):
        result.get_figure("fig3d")
