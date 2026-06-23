"""Tests for custom-target registration behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from qubex.experiment.experiment_context import ExperimentContext
from qubex.system import Mux, Qubit, Resonator, Target, TargetRegistry, TargetType
from qubex.system.control_system import (
    CapChannel,
    CapPort,
    GenChannel,
    GenPort,
    PortType,
)


class _BackendControllerStub:
    def __init__(self) -> None:
        self.define_calls: list[dict[str, object]] = []

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency_ghz: float | None = None,
    ) -> None:
        self.define_calls.append(
            {
                "target_name": target_name,
                "channel_name": channel_name,
                "target_frequency_ghz": target_frequency_ghz,
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
        qubits: Mapping[str, Qubit],
        resonator: Resonator,
        mux: Mux,
    ) -> None:
        self.control_system = control_system
        self.target_registry = target_registry
        self._qubits = dict(qubits)
        self._qubit = self._qubits["Q00"]
        self._resonator = resonator
        self._mux = mux
        self.added_targets: list[Target] = []

    def add_target(self, target: Target) -> None:
        self.added_targets.append(target)
        self.target_registry.register(target)

    def get_qubit(self, label: str) -> Qubit:
        if label not in self._qubits:
            raise KeyError(label)
        return self._qubits[label]

    def get_resonator(self, label: str) -> Resonator:
        if label != self._resonator.label:
            raise KeyError(label)
        return self._resonator

    def get_mux_by_qubit(self, label: str) -> Mux:
        if label not in self._qubits:
            raise KeyError(label)
        return self._mux

    def resolve_cr_pair(self, label: str) -> tuple[str, str]:
        return self.target_registry.resolve_cr_pair(label)

    def resolve_2q_qubits(self, label: str) -> tuple[str, str]:
        return self.target_registry.resolve_2q_qubits(label)


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
    *,
    port: GenPort | CapPort | None = None,
    include_cr_pair: bool = False,
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
    target_qubit = Qubit(
        index=1,
        label="Q01",
        chip_id="chip",
        resonator="RQ01",
        _bare_frequency=5.1,
        _anharmonicity=-0.3,
        _control_frequency_ge=5.1,
        _control_frequency_ef=4.8,
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
    gen_targets: dict[str, Target] = {base_target.label: base_target}
    if include_cr_pair:
        cr_target = Target.new_cr_target(
            control_qubit=qubit,
            target_qubit=target_qubit,
            channel=registry_port.channels[0],
        )
        gen_targets[cr_target.label] = cr_target
    target_registry = TargetRegistry(gen_targets=gen_targets)
    experiment_system = _ExperimentSystemStub(
        control_system=_ControlSystemStub(control_port),
        target_registry=target_registry,
        qubits={qubit.label: qubit, target_qubit.label: target_qubit},
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


def test_register_custom_target_updates_backend_settings_with_official_keyword() -> (
    None
):
    """Given update_backend_settings, when registering target, then backend settings are pushed."""
    context, _, _, system_manager = _make_context()

    context.register_custom_target(
        label="CUSTOM",
        frequency=5.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        qubit_label="Q00",
        update_backend_settings=True,
    )

    assert system_manager.push_calls == [["B0"]]


def test_register_custom_target_preserves_update_lsi_alias() -> None:
    """Given update_lsi, when registering target, then legacy backend update behavior is preserved."""
    context, _, _, system_manager = _make_context()

    with pytest.warns(DeprecationWarning, match="`update_lsi` is deprecated"):
        context.register_custom_target(
            label="CUSTOM",
            frequency=5.2,
            box_id="B0",
            port_number=2,
            channel_number=0,
            qubit_label="Q00",
            update_lsi=True,
        )

    assert system_manager.push_calls == [["B0"]]


def test_register_custom_target_rejects_conflicting_backend_update_keywords() -> None:
    """Given both update keywords, when registering target, then ValueError is raised."""
    context, _, _, _ = _make_context()

    with (
        pytest.warns(DeprecationWarning, match="`update_lsi` is deprecated"),
        pytest.raises(ValueError, match="`update_lsi` conflicts"),
    ):
        context.register_custom_target(
            label="CUSTOM",
            frequency=5.2,
            box_id="B0",
            port_number=2,
            channel_number=0,
            qubit_label="Q00",
            update_backend_settings=True,
            update_lsi=False,
        )


def test_register_custom_target_rejects_unexpected_deprecated_options() -> None:
    """Given unknown keyword option, when registering target, then TypeError is raised."""
    context, _, _, _ = _make_context()

    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        context.register_custom_target(
            label="CUSTOM",
            frequency=5.2,
            box_id="B0",
            port_number=2,
            channel_number=0,
            qubit_label="Q00",
            unexpected_option=True,
        )


@pytest.mark.parametrize(
    ("target_type", "expected_attr"),
    [
        (TargetType.CTRL_2Q, "_qubit"),
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
        target_qubit_label=("Q01" if target_type == TargetType.CTRL_2Q else None),
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


def test_cr_pair_prefers_target_registry_mapping() -> None:
    """Given registered CR label, when resolving pair, then target registry mapping is returned."""
    context, _, _, _ = _make_context(include_cr_pair=True)

    assert context.cr_pair("Q00-Q01") == ("Q00", "Q01")


def test_cr_pair_rejects_unregistered_label_without_parsing() -> None:
    """Given unregistered CR label, when resolving pair, then label parsing is not used."""
    context, _, _, _ = _make_context()

    with pytest.raises(ValueError, match="CR target `Q00-Q01` is not registered"):
        context.cr_pair("Q00-Q01")


def test_resolve_2q_qubits_resolves_custom_bswap_target() -> None:
    """Given a custom bSWAP target, when resolving qubits, then metadata is used."""
    context, experiment_system, _, _ = _make_context()

    context.register_custom_target(
        label="CUSTOM-BSWAP",
        frequency=5.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        target_type=TargetType.CTRL_2Q,
        qubit_label="Q00",
        target_qubit_label="Q01",
        metadata={"gate": "BSWAP"},
    )

    target = experiment_system.added_targets[0]
    assert target.metadata == {
        "gate": "BSWAP",
        "active_qubit": "Q00",
        "passive_qubit": "Q01",
    }
    assert context.resolve_2q_qubits("CUSTOM-BSWAP") == ("Q00", "Q01")


def test_resolve_2q_qubits_resolves_generic_2q_target() -> None:
    """Given a generic custom 2Q target, when resolving qubits, then metadata is used."""
    context, experiment_system, _, _ = _make_context()

    context.register_custom_target(
        label="CUSTOM-2Q",
        frequency=5.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        target_type=TargetType.CTRL_2Q,
        qubit_label="Q00",
        target_qubit_label="Q01",
    )

    target = experiment_system.added_targets[0]
    assert target.metadata == {"qubits": ("Q00", "Q01")}
    assert context.resolve_2q_qubits("CUSTOM-2Q") == ("Q00", "Q01")


def test_register_custom_2q_target_requires_target_qubit_label() -> None:
    """Given custom 2Q target, when target qubit is omitted, then ValueError is raised."""
    context, _, _, _ = _make_context()

    with pytest.raises(ValueError, match="target_qubit_label must be provided"):
        context.register_custom_target(
            label="CUSTOM-2Q",
            frequency=5.2,
            box_id="B0",
            port_number=2,
            channel_number=0,
            target_type=TargetType.CTRL_2Q,
            qubit_label="Q00",
        )
