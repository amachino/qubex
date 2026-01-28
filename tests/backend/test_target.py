"""Tests for backend target helpers."""

from __future__ import annotations

from typing import Literal

import pytest

from qubex.backend import Mux, Qubit, Resonator, Target, TargetType
from qubex.backend.control_system import GenChannel, GenPort, PortType


def _make_gen_channel(*, sideband: Literal["U", "L"]) -> GenChannel:
    port = GenPort(
        id="B0-P0",
        box_id="B0",
        number=0,
        type=PortType.CTRL,
        channels=(),
        sideband=sideband,
        lo_freq=9_000_000_000,
        cnco_freq=1_000_000_000,
    )
    return GenChannel(id="B0-P0-CH0", number=0, _port=port, fnco_freq=0)


def _make_qubit(label: str) -> Qubit:
    return Qubit(
        index=0,
        label=label,
        chip_id="chip",
        resonator=f"R{label}",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
    )


def _make_resonator(label: str, qubit: str) -> Resonator:
    return Resonator(
        index=0,
        label=label,
        chip_id="chip",
        qubit=qubit,
        _readout_frequency=6.5,
    )


def test_target_label_helpers():
    """Target label helpers should parse and format labels."""
    assert Target.qubit_label("Q00") == "Q00"
    assert Target.qubit_label("Q00-ef") == "Q00"
    assert Target.qubit_label("Q00-CR") == "Q00"
    assert Target.qubit_label("Q00-Q01") == "Q00"
    assert Target.qubit_label("Q00_read") == "Q00"
    assert Target.read_label("Q00") == "RQ00"
    assert Target.cr_label("Q00", "Q01") == "Q00-Q01"
    with pytest.raises(ValueError, match="Invalid target label"):
        Target.qubit_label("BAD")


def test_target_frequency_and_availability():
    """Target should compute AWG frequency and availability."""
    qubit = _make_qubit("Q00")
    channel = _make_gen_channel(sideband="U")
    target = Target.new_ge_target(qubit=qubit, channel=channel)
    target.frequency = 9.95  # GHz
    assert target.awg_frequency == pytest.approx(-0.05, abs=1e-6)
    assert target.is_available is True


def test_target_related_qubits_and_qubit_property():
    """Target should resolve related qubits for objects."""
    qubit = _make_qubit("Q00")
    resonator = _make_resonator("RQ00", "Q00")
    mux = Mux(index=0, label="M0", chip_id="chip", resonators=(resonator,))
    channel = _make_gen_channel(sideband="L")

    qubit_target = Target.new_ge_target(qubit=qubit, channel=channel)
    read_target = Target.new_read_target(resonator=resonator, channel=channel)
    pump_target = Target.new_pump_target(mux=mux, frequency=9.0, channel=channel)

    assert qubit_target.qubit == "Q00"
    assert read_target.qubit == "Q00"
    assert pump_target.qubit == ""
    assert qubit_target.is_related_to_qubits(["Q00"]) is True
    assert read_target.is_related_to_qubits(["Q00"]) is True
    assert pump_target.is_related_to_qubits(["Q00"]) is True
    assert qubit_target.type == TargetType.CTRL_GE
