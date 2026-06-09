"""Tests for QuEL-1 port configuration helpers."""

from __future__ import annotations

from qubex.system.control_system import Box
from qubex.system.quantum_system import Qubit
from qubex.system.quel1.quel1_port_configurator import (
    MixingUtil,
    create_control_configuration,
    get_boxes_to_configure,
)


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


def test_ge_ef_fh_two_channel_layout_shares_ef_and_fh_channel() -> None:
    """Given two control channels, ge-ef-fh shares channel 1 between EF and FH."""
    qubit = Qubit(
        index=0,
        label="Q0",
        chip_id="chip",
        resonator="R0",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
        _control_frequency_fh=4.4,
    )

    config = create_control_configuration(
        mode="ge-ef-fh",
        qubit=qubit,
        n_channels=2,
        get_spectator_qubits=lambda _: [],
        excluded_targets=[],
        ssb=None,
    )

    expected_fnco, _ = MixingUtil.calc_fnco(
        f=((qubit.control_frequency_ef + qubit.control_frequency_fh) / 2) * 1e9,
        ssb=None,
        lo=config["lo"],
        cnco=config["cnco"],
    )
    assert config["channels"][0]["targets"] == ["Q0"]
    assert config["channels"][1]["targets"] == ["Q0-ef", "Q0-fh"]
    assert config["channels"][1]["fnco"] == expected_fnco


def test_ge_ef_fh_three_channel_layout_keeps_dedicated_fh_channel() -> None:
    """Given three channels, ge-ef-fh keeps GE, EF, and FH separate."""
    qubit = Qubit(
        index=0,
        label="Q0",
        chip_id="chip",
        resonator="R0",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
        _control_frequency_fh=4.4,
    )

    config = create_control_configuration(
        mode="ge-ef-fh",
        qubit=qubit,
        n_channels=3,
        get_spectator_qubits=lambda _: [],
        excluded_targets=[],
        ssb=None,
    )

    assert config["channels"][0]["targets"] == ["Q0"]
    assert config["channels"][1]["targets"] == ["Q0-ef"]
    assert config["channels"][2]["targets"] == ["Q0-fh"]
    assert config["channels"][1]["fnco"] != config["channels"][2]["fnco"]
