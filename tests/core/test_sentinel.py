"""Tests for shared sentinel helpers."""

from __future__ import annotations

from qubex.core.sentinel import MISSING, make_sentinel


def test_missing_uses_the_configured_name() -> None:
    """Given the shared missing sentinel, when rendered, then it uses the configured name."""
    assert repr(MISSING) == "<MISSING>"


def test_make_sentinel_returns_unique_instances() -> None:
    """Given repeated sentinel creation, when make_sentinel is called, then each sentinel is unique."""
    left = make_sentinel("LEFT")
    right = make_sentinel("LEFT")

    assert left is not right
