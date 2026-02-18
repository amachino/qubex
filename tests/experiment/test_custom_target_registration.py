"""Tests for custom-target registration behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.backend import Mux, Qubit, Resonator, Target, TargetRegistry, TargetType
from qubex.backend.control_system import (
    CapChannel,
    CapPort,
    GenChannel,
    GenPort,
    PortType,
)
from qubex.experiment.experiment_context import ExperimentContext


class _BackendControllerStub:
    def __init__(self) -> None:
        self.define_calls: list[dict[str, object]] = []

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency: float | None = None,
    ) -> None:
        self.define_calls.append(
            {
                "target_name": target_name,
                "channel_name": channel_name,
                "target_frequency": target_frequency,
            }
        )


class _SystemManagerStub:
    def __init__(self) -> None:
        self.push_calls: list[list[str]] = []

    def push(self, *, box_ids: list[str]) -> None:
        self.push_calls.append(box_ids)


@dataclass
class _ControlSystemStub:
    port: GenPort | CapPort

    def get_port(self, box_id: str, port_number: int) -> GenPort | CapPort:
        assert box_id == self.port.box_id
        assert port_number == self.port.number
        return self.port


class _ExperimentSystemStub:
    def __init__(
        self,
        *,
        control_system: _ControlSystemStub,
        target_registry: TargetRegistry,
        qubit: Qubit,
        resonator: Resonator,
        mux: Mux,
    ) -> None:
        self.control_system = control_system
        self.target_registry = target_registry
        self._qubit = qubit
        self._resonator = resonator
        self._mux = mux
        self.added_targets: list[Target] = []

    def add_target(self, target: Target) -> None:
        self.added_targets.append(target)

    def get_qubit(self, label: str) -> Qubit:
        if label != self._qubit.label:
            raise KeyError(label)
        return self._qubit

    def get_resonator(self, label: str) -> Resonator:
        if label != self._resonator.label:
            raise KeyError(label)
        return self._resonator

    def get_mux_by_qubit(self, label: str) -> Mux:
        if label != self._qubit.label:
            raise KeyError(label)
        return self._mux


class _TestExperimentContext(ExperimentContext):
    def __init__(
        self,
        *,
        experiment_system: _ExperimentSystemStub,
        backend_controller: _BackendControllerStub,
        system_manager: _SystemManagerStub,
    ) -> None:
        self._experiment_system_stub = experiment_system
        self._backend_controller_stub = backend_controller
        self._system_manager_stub = system_manager

    @property
    def experiment_system(self) -> _ExperimentSystemStub:
        return self._experiment_system_stub

    @property
    def control_system(self) -> _ControlSystemStub:
        return self._experiment_system_stub.control_system

    @property
    def backend_controller(self) -> _BackendControllerStub:
        return self._backend_controller_stub

    @property
    def system_manager(self) -> _SystemManagerStub:
        return self._system_manager_stub


def _make_gen_port(*, box_id: str = "B0", port_number: int = 2) -> GenPort:
    port = GenPort(
        id=f"{box_id}.CTRL0",
        box_id=box_id,
        number=port_number,
        type=PortType.CTRL,
        channels=(),
        sideband="L",
        lo_freq=10_000_000_000,
        cnco_freq=1_000_000_000,
    )
    channel = GenChannel(id=f"{port.id}.CH0", number=0, _port=port, fnco_freq=0)
    port.channels = (channel,)
    return port


def _make_cap_port(*, box_id: str = "B0", port_number: int = 2) -> CapPort:
    port = CapPort(
        id=f"{box_id}.READ0.IN",
        box_id=box_id,
        number=port_number,
        type=PortType.READ_IN,
        channels=(),
        lo_freq=10_000_000_000,
        cnco_freq=1_000_000_000,
    )
    channel = CapChannel(id=f"{port.id}.CH0", number=0, _port=port, fnco_freq=0)
    port.channels = (channel,)
    return port


def _make_context(
    *, port: GenPort | CapPort | None = None
) -> tuple[
    _TestExperimentContext,
    _ExperimentSystemStub,
    _BackendControllerStub,
    _SystemManagerStub,
]:
    qubit = Qubit(
        index=0,
        label="Q00",
        chip_id="chip",
        resonator="RQ00",
        _bare_frequency=5.0,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.0,
        _control_frequency_ef=4.7,
    )
    resonator = Resonator(
        index=0,
        label="RQ00",
        chip_id="chip",
        qubit="Q00",
        _readout_frequency=6.5,
    )
    mux = Mux(index=0, label="M0", chip_id="chip", resonators=(resonator,))
    control_port = port or _make_gen_port()
    registry_port = _make_gen_port()
    base_target = Target.new_ge_target(qubit=qubit, channel=registry_port.channels[0])
    target_registry = TargetRegistry(gen_targets={base_target.label: base_target})
    experiment_system = _ExperimentSystemStub(
        control_system=_ControlSystemStub(control_port),
        target_registry=target_registry,
        qubit=qubit,
        resonator=resonator,
        mux=mux,
    )
    backend_controller = _BackendControllerStub()
    system_manager = _SystemManagerStub()
    context = _TestExperimentContext(
        experiment_system=experiment_system,
        backend_controller=backend_controller,
        system_manager=system_manager,
    )
    return context, experiment_system, backend_controller, system_manager


def test_register_custom_target_requires_qubit_label_for_unresolved_label() -> None:
    """Given unresolved custom label, when qubit_label is omitted, then ValueError is raised."""
    context, _, _, _ = _make_context()

    with pytest.raises(ValueError, match="Pass `qubit_label` explicitly"):
        context.register_custom_target(
            label="CUSTOM",
            frequency=5.1,
            box_id="B0",
            port_number=2,
            channel_number=0,
        )


def test_register_custom_target_registers_with_explicit_qubit_label() -> None:
    """Given explicit qubit label, when registering target, then backend and system are updated."""
    context, experiment_system, backend_controller, _ = _make_context()

    context.register_custom_target(
        label="CUSTOM",
        frequency=5.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        qubit_label="Q00",
    )

    assert len(backend_controller.define_calls) == 1
    assert backend_controller.define_calls[0]["target_name"] == "CUSTOM"
    assert len(experiment_system.added_targets) == 1
    assert experiment_system.added_targets[0].label == "CUSTOM"
    assert experiment_system.added_targets[0].qubit == "Q00"


@pytest.mark.parametrize(
    ("target_type", "expected_attr"),
    [
        (TargetType.READ, "_resonator"),
        (TargetType.PUMP, "_mux"),
    ],
)
def test_register_custom_target_selects_physical_object_by_target_type(
    target_type: TargetType,
    expected_attr: str,
) -> None:
    """Given target type, when registering custom target, then physical object is selected accordingly."""
    context, experiment_system, _, _ = _make_context()

    context.register_custom_target(
        label=f"CUSTOM-{target_type.value}",
        frequency=5.3,
        box_id="B0",
        port_number=2,
        channel_number=0,
        target_type=target_type,
        qubit_label="Q00",
    )

    assert len(experiment_system.added_targets) == 1
    expected_object = getattr(experiment_system, expected_attr)
    assert experiment_system.added_targets[0].object is expected_object


def test_register_custom_target_rejects_non_generator_port() -> None:
    """Given non-generator port, when registering custom target, then TypeError is raised."""
    cap_port = _make_cap_port()
    context, _, _, _ = _make_context(port=cap_port)

    with pytest.raises(TypeError, match="must be a GenPort"):
        context.register_custom_target(
            label="Q00-CUSTOM",
            frequency=5.2,
            box_id="B0",
            port_number=2,
            channel_number=0,
            qubit_label="Q00",
        )
