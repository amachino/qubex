"""Tests for QuEL-1 port configuration helpers."""

from __future__ import annotations

from qubex.system.control_system import Box
from qubex.system.quel1.quel1_port_configurator import get_boxes_to_configure


def test_get_boxes_to_configure_selects_only_quel1_family_boxes() -> None:
    """Given mixed boxes, when selecting configurable boxes, then only QuEL-1 family boxes are returned."""
    quel1_box = Box.new(
        id="Q1",
        name="QuEL-1",
        type="quel1-a",
        address="192.0.2.10",
        adapter="A1",
    )
    quel3_box = Box.new(
        id="Q3",
        name="QuEL-3",
        type="quel3",
        address="192.0.2.30",
        adapter="A3",
    )

    assert get_boxes_to_configure([quel1_box, quel3_box]) == [quel1_box]
