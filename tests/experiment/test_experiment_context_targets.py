"""Tests for ExperimentContext target filtering."""

from __future__ import annotations

from typing import Literal

from qubex.experiment.experiment_context import ExperimentContext
from qubex.system import Qubit, Target
from qubex.system.control_system import GenChannel, GenPort, PortType


class _ExperimentSystemStub:
    def __init__(
        self,
        *,
        targets: list[Target],
        cr_pairs: dict[str, tuple[str, str]],
    ) -> None:
        self.targets = targets
        self._cr_pairs = cr_pairs

    def resolve_cr_pair(self, label: str) -> tuple[str, str]:
        return self._cr_pairs[label]


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


def _make_qubit(label: str, frequency: float) -> Qubit:
    return Qubit(
        index=int(label[1:]),
        label=label,
        chip_id="chip",
        resonator=f"R{label}",
        _bare_frequency=frequency,
        _anharmonicity=-0.3,
        _control_frequency_ge=frequency,
        _control_frequency_ef=frequency - 0.3,
    )


def test_targets_exclude_cr_pair_with_inactive_spectator(monkeypatch) -> None:
    """Given inactive spectator, when listing targets, then pair CR target is excluded."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_qubits"] = ["Q00"]

    channel = _make_gen_channel()
    control_qubit = _make_qubit("Q00", 5.0)
    spectator_qubit = _make_qubit("Q01", 5.1)
    ge_target = Target.new_ge_target(qubit=control_qubit, channel=channel)
    default_cr_target = Target.new_cr_target(
        control_qubit=control_qubit,
        channel=channel,
    )
    pair_cr_target = Target.new_cr_target(
        control_qubit=control_qubit,
        target_qubit=spectator_qubit,
        channel=channel,
    )
    experiment_system = _ExperimentSystemStub(
        targets=[ge_target, default_cr_target, pair_cr_target],
        cr_pairs={
            default_cr_target.label: ("Q00", "CR"),
            pair_cr_target.label: ("Q00", "Q01"),
        },
    )

    monkeypatch.setattr(
        ExperimentContext,
        "experiment_system",
        property(lambda self: experiment_system),
    )

    assert set(context.targets) == {"Q00", "Q00-CR"}


def test_targets_keep_cr_pair_with_active_spectator(monkeypatch) -> None:
    """Given active spectator, when listing targets, then pair CR target is included."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_qubits"] = ["Q00", "Q01"]

    channel = _make_gen_channel()
    control_qubit = _make_qubit("Q00", 5.0)
    spectator_qubit = _make_qubit("Q01", 5.1)
    pair_cr_target = Target.new_cr_target(
        control_qubit=control_qubit,
        target_qubit=spectator_qubit,
        channel=channel,
    )
    experiment_system = _ExperimentSystemStub(
        targets=[pair_cr_target],
        cr_pairs={pair_cr_target.label: ("Q00", "Q01")},
    )

    monkeypatch.setattr(
        ExperimentContext,
        "experiment_system",
        property(lambda self: experiment_system),
    )

    assert set(context.targets) == {"Q00-Q01"}
