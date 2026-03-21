"""Tests for the experiment Result model."""

from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go
import pytest

from qubex.experiment.models.result import Result


def test_result_should_preserve_mapping_payload_access() -> None:
    """Result should preserve mapping access for payload data."""
    result = Result(data={"value": 1, "label": "Q00"})

    assert result.data == {"value": 1, "label": "Q00"}
    assert result["value"] == 1
    assert result["label"] == "Q00"
    assert dict(result) == {"value": 1, "label": "Q00"}
    assert datetime.fromisoformat(result.created_at).tzinfo == timezone.utc
    assert result.figure is None
    assert result.figures is None


def test_result_should_not_derive_figure_fields_from_legacy_payload_keys() -> None:
    """Result should not derive figure attributes from legacy payload keys."""
    figure = go.Figure()
    figures = {"Q00": go.Figure()}

    result = Result(
        data={
            "fig": figure,
            "figures": figures,
        }
    )

    assert result.figure is None
    assert result.figures is None


def test_result_should_warn_on_legacy_figure_key_access() -> None:
    """Result should warn when legacy figure payload keys are accessed."""
    figure = go.Figure()
    figures = {"Q00": go.Figure()}
    result = Result(
        data={
            "figure": figure,
            "figures": figures,
        }
    )

    with pytest.warns(DeprecationWarning, match="figure` attribute"):
        assert result["figure"] is figure

    with pytest.warns(DeprecationWarning, match="figures` attribute"):
        assert result.data["figures"] == figures


def test_result_should_prefer_explicit_figure_fields_over_legacy_payload_keys() -> None:
    """Result should prefer explicit figure fields over legacy payload keys."""
    explicit_figure = go.Figure()
    explicit_figures = {"explicit": go.Figure()}
    legacy_figure = go.Figure()
    legacy_figures = {"legacy": go.Figure()}

    result = Result(
        data={
            "fig": legacy_figure,
            "figures": legacy_figures,
        },
        figure=explicit_figure,
        figures=explicit_figures,
    )

    assert result.figure is explicit_figure
    assert result.figures == explicit_figures
    with pytest.warns(DeprecationWarning, match="figure` attribute"):
        assert result["fig"] is legacy_figure
    with pytest.warns(DeprecationWarning, match="figures` attribute"):
        assert result["figures"] == legacy_figures


def test_result_should_return_primary_and_named_figures_via_get_figure() -> None:
    """Result should return primary and named figures via get_figure."""
    figure = go.Figure()
    figures = {"overview": go.Figure(), "detail": go.Figure()}
    result = Result(
        data={"value": 1},
        figure=figure,
        figures=figures,
    )

    assert result.get_figure() is figure
    assert result.get_figure("overview") is figures["overview"]


def test_result_should_raise_when_requested_figure_is_missing() -> None:
    """Result should raise when the requested figure is missing."""
    result = Result(data={"value": 1})

    with pytest.raises(ValueError, match="primary figure"):
        result.get_figure()

    with pytest.raises(KeyError, match="overview"):
        result.get_figure("overview")
