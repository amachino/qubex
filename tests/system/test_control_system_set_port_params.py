"""Tests for `ControlSystem.set_port_params` behavior."""

from __future__ import annotations

from qubex.core.sentinel import MISSING
from qubex.system.control_system import Box, ControlSystem


def test_set_port_params_preserves_cap_lo_freq_when_not_given() -> None:
    """Given only rfswitch for CapPort, when setting params, then lo_freq remains unchanged."""
    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="none",
        port_numbers=[0],
    )
    control_system = ControlSystem([box], clock_master_address="192.0.2.1")
    cap_port = control_system.get_cap_port("B0", 0)
    original_lo_freq = cap_port.lo_freq

    control_system.set_port_params(
        box_id="B0",
        port_number=0,
        rfswitch="loop",
    )

    assert cap_port.rfswitch == "loop"
    assert cap_port.lo_freq == original_lo_freq


def test_set_port_params_accepts_shared_missing_sentinel() -> None:
    """Given the shared missing sentinel, when setting params, then omitted values remain unchanged."""
    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="none",
        port_numbers=[0],
    )
    control_system = ControlSystem([box], clock_master_address="192.0.2.1")
    cap_port = control_system.get_cap_port("B0", 0)
    original_lo_freq = cap_port.lo_freq

    control_system.set_port_params(
        box_id="B0",
        port_number=0,
        lo_freq=MISSING,
    )

    assert cap_port.lo_freq == original_lo_freq
