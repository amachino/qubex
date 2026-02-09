"""Tests for control system option-driven channel mapping."""

from __future__ import annotations

import pytest

from qubex.backend.control_system import Box


def _control_channel_counts(box: Box) -> list[int]:
    """Return channel counts for R8 control ports 6-9."""
    return [len(box.get_port(port_num).channels) for port_num in (6, 7, 8, 9)]


def test_r8_box_uses_default_awg2222_when_options_omitted() -> None:
    """Given R8 box without options, when building ports, then control channels are 2-2-2-2."""
    box = Box.new(
        id="B0",
        name="R8",
        type="quel1se-riken8",
        address="192.0.2.10",
        adapter="A0",
    )

    assert _control_channel_counts(box) == [2, 2, 2, 2]


def test_r8_box_applies_awg1331_option_to_control_channels() -> None:
    """Given R8 box with awg1331 option, when building ports, then control channels are 1-3-3-1."""
    box = Box.new(
        id="B0",
        name="R8",
        type="quel1se-riken8",
        address="192.0.2.10",
        adapter="A0",
        options=("se8_mxfe1_awg1331",),
    )

    assert _control_channel_counts(box) == [1, 3, 3, 1]


def test_r8_box_rejects_multiple_awg_options() -> None:
    """Given R8 box with conflicting awg options, when building ports, then ValueError is raised."""
    with pytest.raises(ValueError, match="Multiple AWG options are not allowed"):
        Box.new(
            id="B0",
            name="R8",
            type="quel1se-riken8",
            address="192.0.2.10",
            adapter="A0",
            options=("se8_mxfe1_awg1331", "se8_mxfe1_awg2222"),
        )
