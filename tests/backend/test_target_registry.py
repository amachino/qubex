"""Tests for backend target registry helpers."""

from __future__ import annotations

from typing import Literal, cast

import pytest

from qubex.backend import Qubit, Resonator, Target, TargetType
from qubex.backend.control_system import (
    CapChannel,
    CapPort,
    GenChannel,
    GenPort,
    PortType,
)
from qubex.backend.target import CapTarget
from qubex.backend.target_registry import TargetRegistry


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


def _make_gen_channel(*, sideband: Literal["U", "L"] = "U") -> GenChannel:
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


def _make_cap_channel() -> CapChannel:
    port = CapPort(
        id="B0-P1",
        box_id="B0",
        number=1,
        type=PortType.READ_IN,
        channels=(),
        lo_freq=9_000_000_000,
        cnco_freq=1_000_000_000,
    )
    return CapChannel(id="B0-P1-CH0", number=0, _port=port, fnco_freq=0)


def _build_registry() -> TargetRegistry:
    qubit0 = _make_qubit("Q00")
    qubit1 = _make_qubit("Q01")
    read0 = _make_resonator("RQ00", "Q00")
    read1 = _make_resonator("RQ01", "Q01")
    gen_channel = _make_gen_channel()

    ge0 = Target.new_ge_target(qubit=qubit0, channel=gen_channel)
    ef0 = Target.new_ef_target(qubit=qubit0, channel=gen_channel)
    readout0 = Target.new_read_target(resonator=read0, channel=gen_channel)
    readout1 = Target.new_read_target(resonator=read1, channel=gen_channel)
    cr_default = Target.new_cr_target(control_qubit=qubit0, channel=gen_channel)
    cr_pair = Target.new_cr_target(
        control_qubit=qubit0,
        target_qubit=qubit1,
        channel=gen_channel,
    )
    cap_read0 = CapTarget.new_read_target(
        resonator=read0,
        channel=_make_cap_channel(),
    )

    return TargetRegistry(
        gen_targets={
            ge0.label: ge0,
            ef0.label: ef0,
            readout0.label: readout0,
            readout1.label: readout1,
            cr_default.label: cr_default,
            cr_pair.label: cr_pair,
        },
        cap_targets={cap_read0.label: cap_read0},
    )


def test_target_registry_resolves_qubit_and_phase_labels() -> None:
    """Given mixed labels, when resolving labels, then qubit/GE/EF/read labels are returned."""
    registry = _build_registry()

    assert registry.resolve_qubit_label("Q00") == "Q00"
    assert registry.resolve_qubit_label("Q00-ef") == "Q00"
    assert registry.resolve_qubit_label("RQ00") == "Q00"
    assert registry.resolve_ge_label("RQ00") == "Q00"
    assert registry.resolve_ef_label("RQ00") == "Q00-ef"
    assert registry.resolve_read_label("Q00-ef") == "RQ00"


def test_target_registry_resolves_cr_labels() -> None:
    """Given registered CR targets, when resolving, then default and pair labels are returned."""
    registry = _build_registry()

    assert registry.resolve_cr_label("Q00") == "Q00-CR"
    assert registry.resolve_cr_label("Q00", "Q01") == "Q00-Q01"
    assert registry.resolve_cr_label("Q00-CR") == "Q00-CR"
    assert registry.resolve_cr_pair("Q00-CR") == ("Q00", "CR")
    assert registry.resolve_cr_pair("Q00-Q01") == ("Q00", "Q01")


def test_target_registry_legacy_resolution_is_opt_in() -> None:
    """Given parser-compatible labels, when legacy flag is enabled, then fallback resolution works."""
    registry = _build_registry()

    with pytest.raises(ValueError, match="Qubit label could not be resolved"):
        registry.resolve_qubit_label("Q00_read")

    assert registry.resolve_qubit_label("Q00_read", allow_legacy=True) == "Q00"
    assert registry.resolve_ge_label("Q00_read", allow_legacy=True) == "Q00"
    assert registry.resolve_ef_label("Q00_read", allow_legacy=True) == "Q00-ef"
    assert registry.resolve_read_label("Q00_read", allow_legacy=True) == "RQ00"
    assert registry.resolve_cr_pair("Q00-Q01", allow_legacy=True) == ("Q00", "Q01")


def test_target_registry_measurement_output_label_prefers_qubit() -> None:
    """Given target labels, when mapping measurement output, then qubit labels are preferred."""
    registry = _build_registry()

    assert registry.measurement_output_label("RQ00") == "Q00"
    assert registry.measurement_output_label("Q00") == "Q00"
    assert registry.measurement_output_label("M0") == "M0"


def test_target_registry_get_returns_target_metadata() -> None:
    """Given registered labels, when fetching metadata, then resolved kind and qubit are returned."""
    registry = _build_registry()

    ge = registry.get("Q00")
    read = registry.get("RQ00")

    assert ge.kind == "gen"
    assert ge.qubit_label == "Q00"
    assert read.kind == "gen"
    assert read.qubit_label == "Q00"


def test_target_registry_raises_for_unknown_labels() -> None:
    """Given unknown labels, when resolving, then descriptive errors are raised."""
    registry = _build_registry()
    qubit = _make_qubit("Q10")
    ge = Target.new_ge_target(qubit=qubit, channel=_make_gen_channel())
    no_read_registry = TargetRegistry(gen_targets={ge.label: ge})

    with pytest.raises(ValueError, match="Qubit label could not be resolved"):
        registry.resolve_qubit_label("BAD")
    with pytest.raises(ValueError, match="Readout target is not registered"):
        no_read_registry.resolve_read_label("Q10")
    with pytest.raises(KeyError, match="Target `BAD` is not registered"):
        registry.get("BAD")


def test_target_registry_rejects_non_target_entries() -> None:
    """Given invalid entries, when building registry, then TypeError is raised."""
    qubit = _make_qubit("Q00")
    ge = Target.new_target(
        label="GEN",
        frequency=5.0,
        object=qubit,
        channel=_make_gen_channel(),
        type=TargetType.UNKNOWN,
    )

    invalid_entries = cast(dict[str, Target], {"GEN": ge, "RAW": object()})
    with pytest.raises(TypeError, match="gen_targets entries must be Target"):
        TargetRegistry(gen_targets=invalid_entries)
