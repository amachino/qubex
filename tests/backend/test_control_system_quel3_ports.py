"""Tests for QuEL-3 control-system port mapping."""

from __future__ import annotations

from qubex.system.control_system import Box, PortType


def test_quel3_box_has_front_panel_port_types() -> None:
    """Given a QuEL-3 box, when creating ports, then numbers 0-17 map to documented types."""
    box = Box.new(
        id="Q3",
        name="QuEL-3",
        type="quel3",
        address="192.0.2.30",
        adapter="A3",
    )

    expected = {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        2: PortType.CTRL,
        3: PortType.PUMP,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.READ_IN,
        8: PortType.READ_OUT,
        9: PortType.CTRL,
        10: PortType.PUMP,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
        14: PortType.CTRL,
        15: PortType.CTRL,
        16: PortType.CTRL,
        17: PortType.CTRL,
    }

    assert [port.number for port in box.ports] == list(range(18))
    for port_number, port_type in expected.items():
        assert box.get_port(port_number).type == port_type


def test_quel3_box_has_documented_channel_counts() -> None:
    """Given a QuEL-3 box, when creating ports, then channel counts follow documented simultaneous bands."""
    box = Box.new(
        id="Q3",
        name="QuEL-3",
        type="quel3",
        address="192.0.2.30",
        adapter="A3",
    )

    expected = {
        0: 8,
        1: 2,
        2: 1,
        3: 1,
        4: 2,
        5: 1,
        6: 1,
        7: 8,
        8: 2,
        9: 1,
        10: 1,
        11: 2,
        12: 1,
        13: 1,
        14: 1,
        15: 1,
        16: 2,
        17: 2,
    }

    for port_number, n_channels in expected.items():
        assert len(box.get_port(port_number).channels) == n_channels


def test_quel3_box_traits_follow_direct_nco_control_profile() -> None:
    """Given a QuEL-3 box, when reading traits, then direct NCO control defaults are used."""
    box = Box.new(
        id="Q3",
        name="QuEL-3",
        type="quel3",
        address="192.0.2.30",
        adapter="A3",
    )

    assert box.traits.ctrl_ssb is None
    assert box.traits.readout_ssb is None
    assert box.traits.readout_cnco_center is None
    assert box.traits.default_control_frequency_range == (3.0, 5.0, 0.005)
