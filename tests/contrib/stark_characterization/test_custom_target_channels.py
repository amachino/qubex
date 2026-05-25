"""Tests for custom-channel APIs in `qubex.contrib.experiment.stark_characterization`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from qubex.contrib import (
    experiment as contrib_experiment,
    insitu_target,
    make_insitu_channel,
    make_stark_channel,
    make_stark_cr_channel,
    stark_cr_target,
    stark_target,
)
from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.system import TargetType


@dataclass
class _PortStub:
    box_id: str = "B0"
    number: int = 2


@dataclass
class _ChannelStub:
    number: int = 1
    port: _PortStub = field(default_factory=_PortStub)


@dataclass
class _TargetStub:
    frequency: float = 5.2
    channel: _ChannelStub = field(default_factory=_ChannelStub)


class _ContextStub:
    def resolve_qubit_label(self, target: str) -> str:
        if target in {"Q17", "Q17-ef"}:
            return "Q17"
        if target in {"Q18", "Q18_insitu"}:
            return "Q18"
        raise ValueError(target)


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = _ContextStub()
        self.targets = {
            "Q17": _TargetStub(frequency=5.2),
            "Q18": _TargetStub(frequency=4.8),
            "Q18_insitu": _TargetStub(frequency=4.75),
            "Q17-Q18": _TargetStub(frequency=4.8),
        }
        self.calls: list[dict[str, Any]] = []

    def register_custom_target(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.targets[kwargs["label"]] = _TargetStub(frequency=kwargs["frequency"])


def test_custom_channel_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then custom-channel helpers are available."""
    assert callable(stark_target)
    assert callable(insitu_target)
    assert callable(make_stark_channel)
    assert callable(make_insitu_channel)
    assert callable(stark_cr_target)
    assert callable(make_stark_cr_channel)
    assert callable(contrib_experiment.stark_target)
    assert callable(contrib_experiment.insitu_target)
    assert callable(contrib_experiment.make_stark_channel)
    assert callable(contrib_experiment.make_insitu_channel)


def test_custom_target_labels_resolve_canonical_qubit_label() -> None:
    """Given aliased target labels, when label helpers run, then canonical qubit labels are used."""
    stub = _ExperimentStub()
    exp = cast(Experiment, stub)

    assert stark_target(exp, "Q17-ef") == "Q17_stark"
    assert insitu_target(exp, "Q17-ef") == "Q17_insitu"
    assert (
        stark_cr_target(exp, "Q17", "Q18", stark_drive_qubit="target")
        == "Q17-Q18_insitu"
    )
    assert (
        stark_cr_target(exp, "Q17", "Q18", stark_drive_qubit="control")
        == "Q17_insitu-Q18"
    )


def test_make_custom_channel_registers_with_explicit_qubit_label() -> None:
    """Given a detuned channel helper, when called, then custom registration gets qubit_label."""
    for register, expected_label in (
        (make_stark_channel, "Q17_stark"),
        (make_insitu_channel, "Q17_insitu"),
    ):
        stub = _ExperimentStub()
        exp = cast(Experiment, stub)

        register(exp, "Q17", detuning=0.25, lsi=True, channel=2)

        assert len(stub.calls) == 1
        call = stub.calls[0]
        assert abs(cast(float, call["frequency"]) - 5.45) < 1e-12
        assert call == {
            "label": expected_label,
            "frequency": call["frequency"],
            "box_id": "B0",
            "port_number": 2,
            "channel_number": 3,
            "qubit_label": "Q17",
            "update_lsi": True,
        }


def test_make_stark_cr_channel_registers_ctrl_cr_target() -> None:
    """Given a dressed CR helper, when called, then it registers a CTRL_CR target."""
    stub = _ExperimentStub()
    exp = cast(Experiment, stub)

    make_stark_cr_channel(
        exp,
        control_qubit="Q17",
        target_qubit="Q18",
        stark_drive_qubit="target",
        detuning=0.01,
        lsi=True,
    )

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call == {
        "label": "Q17-Q18_insitu",
        "frequency": 4.76,
        "box_id": "B0",
        "port_number": 2,
        "channel_number": 1,
        "qubit_label": "Q17",
        "target_type": TargetType.CTRL_CR,
        "update_lsi": True,
    }


def test_ensure_stark_cr_channel_registers_missing_target_once() -> None:
    """Given a missing dressed CR target, when ensured, then registration is idempotent."""
    stub = _ExperimentStub()
    exp = cast(Experiment, stub)
    ensure_stark_cr_channel = sc._ensure_stark_cr_channel  # noqa: SLF001

    label = ensure_stark_cr_channel(
        exp,
        control_qubit="Q17",
        target_qubit="Q18",
        stark_drive_qubit="target",
        cr_frequency=4.76,
        update_lsi=False,
    )
    label_again = ensure_stark_cr_channel(
        exp,
        control_qubit="Q17",
        target_qubit="Q18",
        stark_drive_qubit="target",
        cr_frequency=4.76,
        update_lsi=False,
    )

    assert label == "Q17-Q18_insitu"
    assert label_again == "Q17-Q18_insitu"
    assert len(stub.calls) == 1
