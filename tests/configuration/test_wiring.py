"""Tests for configuration wiring helpers."""

from __future__ import annotations

import pytest

from qubex.system.wiring import split_box_port_specifier


def test_split_box_port_specifier_accepts_hyphenated_box_id() -> None:
    """Given hyphenated box id, when splitting port specifier, then id and port are resolved."""
    box_id, port = split_box_port_specifier("unit-a-11")

    assert box_id == "unit-a"
    assert port == 11


def test_split_box_port_specifier_accepts_colon_separator() -> None:
    """Given colon-separated specifier, when splitting, then id and port are resolved."""
    box_id, port = split_box_port_specifier("unit-a:11")

    assert box_id == "unit-a"
    assert port == 11


def test_split_box_port_specifier_rejects_missing_separator() -> None:
    """Given invalid specifier without separator, when splitting, then ValueError is raised."""
    with pytest.raises(ValueError, match="Invalid port specifier"):
        split_box_port_specifier("unita")


def test_split_box_port_specifier_rejects_non_integer_port() -> None:
    """Given invalid specifier with non-integer port, when splitting, then ValueError is raised."""
    with pytest.raises(ValueError, match="Invalid port number"):
        split_box_port_specifier("unit-a-p1")
