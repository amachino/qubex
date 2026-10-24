"""Tests for custom-target registration behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.backend.quel3.infra import Quel3ResourceReader
from qubex.backend.quel3.managers import Quel3ConfigurationManager
from qubex.experiment.experiment_context import ExperimentContext
from qubex.system import (
    Mux,
    Qubit,
    Resonator,
    SystemManager,
    Target,
    TargetRegistry,
    TargetType,
)
from qubex.system.control_system import (
    Box,
    CapChannel,
    CapPort,
    GenChannel,
    GenPort,
    PortType,
)


class _BackendControllerStub(Quel1BackendController):
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
    manager: SystemManager

    def __init__(self) -> None:
        self.push_calls: list[list[str]] = []

    def push(self, *, box_ids: list[str]) -> None:
        self.push_calls.append(box_ids)

    def register_custom_target(self, **kwargs: Any) -> None:
        self.manager.register_custom_target(**kwargs)

    def resolve_custom_target_channel(self, **kwargs: Any) -> GenChannel:
        return self.manager.resolve_custom_target_channel(**kwargs)


@dataclass
class _ControlSystemStub:
    port: GenPort | CapPort

    def get_port(self, box_id: str, port_number: int) -> GenPort | CapPort:
        assert box_id == self.port.box_id
        assert port_number == self.port.number
        return self.port


class _ExperimentSystemStub:
    control_params: Any
    wiring_info: Any
    get_mux_by_readout_port: Any

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

    @property
    def gen_targets(self) -> dict[str, Target]:
        return self.target_registry.gen_targets

    def get_target(self, label: str) -> Target:
        return self.target_registry.get_gen_target(label)

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
    backend_controller: Any = None,
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
    backend_controller = backend_controller or _BackendControllerStub()
    system_manager = _SystemManagerStub()
    manager = object.__new__(SystemManager)
    manager._experiment_system = cast(Any, experiment_system)  # noqa: SLF001
    manager._backend_controller = cast(Any, backend_controller)  # noqa: SLF001
    manager._system_synchronizer = manager._create_system_synchronizer(  # noqa: SLF001
        backend_controller
    )
    manager.push = cast(Any, system_manager.push)
    system_manager.manager = manager
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
        (TargetType.CTRL_FH, "_qubit"),
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


@pytest.fixture
def quel3_registration(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Provide real QuEL-3 registration and cache management with in-memory hardware IO."""
    box = Box.new(id="Q3", name="QuEL-3", type="quel3", address="", adapter="")
    configuration_manager = Quel3ConfigurationManager()
    resource_reader = Quel3ResourceReader()
    controller = Quel3BackendController(
        configuration_manager=configuration_manager, resource_reader=resource_reader
    )
    # Inspect identity invalidation directly: the public export omits resource IDs.
    cache = controller._instrument_cache  # noqa: SLF001
    context, system, _, manager = _make_context(
        port=box.get_port(2), backend_controller=controller
    )
    hardware = SimpleNamespace(infos=[], calls=[], failure=None, read_count=0)

    def info(alias: str, port_id: str) -> Any:
        return SimpleNamespace(
            id=f"{port_id}:{alias}:{len(hardware.calls)}",
            port_id=port_id,
            definition=SimpleNamespace(
                alias=alias,
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4e9, frequency_range_max=6e9
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000, timeline_step_samples=1),
        )

    hardware.infos = [info("sibling", "Q3:tx_p02"), info("elsewhere", "Q3:tx_p04")]
    cache.replace_all(instrument_infos=hardware.infos)

    async def deploy(*, specifications: Any, append: bool, parallel: bool) -> None:
        assert append is True
        assert len(specifications) == 1
        spec = specifications[0]
        hardware.calls.append(spec)
        hardware.infos = [
            item
            for item in hardware.infos
            if (item.definition.alias, item.port_id) != (spec.alias, spec.port_id)
        ]
        item = info(spec.alias, spec.port_id)
        item.definition.role = spec.role
        item.definition.profile.frequency_range_min = spec.frequency_range_min_hz
        item.definition.profile.frequency_range_max = spec.frequency_range_max_hz
        hardware.infos.append(item)
        if hardware.failure == "write":
            raise RuntimeError("write failed")

    def read(*, port_ids: Any, parallel: bool) -> Any:
        hardware.read_count += 1
        if hardware.failure == "read":
            raise RuntimeError("read failed")
        return tuple(item for item in hardware.infos if item.port_id in port_ids)

    monkeypatch.setattr(configuration_manager, "_deploy_instruments", deploy)
    monkeypatch.setattr(resource_reader, "read_instrument_infos", read)
    system.control_params = SimpleNamespace(get_frequency_margin=lambda _: 0.1)
    system.wiring_info = SimpleNamespace(
        read_in=[(system.get_mux_by_qubit("Q00"), box.get_port(0))]
    )
    system.get_mux_by_readout_port = lambda port: (
        system.get_mux_by_qubit("Q00") if port.number == 1 else None
    )
    return SimpleNamespace(
        context=context,
        system=system,
        controller=controller,
        manager=manager,
        box=box,
        hardware=hardware,
        cache=cache,
    )


def _register_quel3(env: Any, **overrides: Any) -> None:
    options = dict(
        label="custom", frequency=4.2, box_id="Q3", port_number=2, qubit_label="Q00"
    )
    options.update(overrides)
    env.context.register_custom_target(**options)


@pytest.mark.parametrize("update", [None, False, True])
@pytest.mark.parametrize(
    "target_type", [kind for kind in TargetType if kind != TargetType.UNKNOWN]
)
def test_quel3_custom_target_registration(
    quel3_registration: Any, target_type: TargetType, update: bool | None
) -> None:
    """Custom targets should always append an instrument before registration regardless of the update flag."""
    env = quel3_registration
    port_number = (
        1
        if target_type == TargetType.READ
        else 3
        if target_type == TargetType.PUMP
        else 2
    )
    port = env.box.get_port(port_number)
    env.context.control_system.port = port
    _register_quel3(
        env,
        port_number=port_number,
        target_type=target_type,
        target_qubit_label="Q01"
        if target_type in (TargetType.CTRL_CR, TargetType.CTRL_2Q)
        else None,
        **({} if update is None else {"update_backend_settings": update}),
    )
    target = env.system.get_target("custom")
    assert target.channel is port.channels[0]
    assert target.type == target_type
    assert target.frequency == 4.2
    assert port.cnco_freq is None
    assert target.channel.fnco_freq is None
    assert env.manager.push_calls == []
    assert len(env.hardware.calls) == 1
    assert env.hardware.read_count == 1
    cache = env.cache.snapshot()
    assert set(cache) == {"sibling", "elsewhere", "custom"}
    spec = env.hardware.calls[0]
    assert spec.alias == "custom"
    assert spec.port_id == (
        "Q3:trx_p00p01"
        if target_type == TargetType.READ
        else f"Q3:tx_p{port_number:02d}"
    )
    assert spec.role == (
        "TRANSCEIVER" if target_type == TargetType.READ else "TRANSMITTER"
    )
    assert spec.frequency_range_min_hz == pytest.approx(4.1e9)
    assert spec.frequency_range_max_hz == pytest.approx(4.3e9)


@pytest.mark.parametrize("failure", ["write", "read"])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("update", [None, False, True])
def test_quel3_registration_failure_preserves_registry(
    quel3_registration: Any, failure: str, replace: bool, update: bool | None
) -> None:
    """Failed deployment should retain the registry and invalidate only the touched port cache."""
    env = quel3_registration
    if replace:
        _register_quel3(env, update_backend_settings=True)
    previous = env.system.gen_targets
    env.hardware.failure = failure
    with pytest.raises(RuntimeError, match="failed"):
        _register_quel3(
            env,
            frequency=4.4,
            **({} if update is None else {"update_backend_settings": update}),
        )
    assert env.system.gen_targets == previous
    assert set(env.cache.snapshot()) == {"elsewhere"}


def test_quel3_same_port_reregistration_refreshes_identity(
    quel3_registration: Any,
) -> None:
    """Re-registering one alias should replace its frequency and cached identity while preserving siblings."""
    env = quel3_registration
    _register_quel3(env, update_backend_settings=True)
    before = env.cache.get("custom").id
    _register_quel3(env, frequency=4.5, update_backend_settings=True)
    assert env.cache.get("custom").id != before
    assert env.system.get_target("custom").frequency == 4.5
    assert set(env.cache.snapshot()) == {"custom", "sibling", "elsewhere"}


@pytest.mark.parametrize(
    ("options", "exception", "match"),
    [
        ({"label": " bad "}, ValueError, "alias"),
        ({"label": ""}, ValueError, "alias"),
        ({"label": "unit:bad"}, ValueError, "alias"),
        ({"target_type": TargetType.UNKNOWN}, ValueError, "Unsupported target type"),
        ({"frequency": float("nan")}, ValueError, "finite"),
        ({"frequency": float("inf")}, ValueError, "finite"),
        ({"channel_number": -1}, IndexError, "out of range"),
        ({"channel_number": 1}, IndexError, "out of range"),
    ],
)
def test_quel3_invalid_registration_is_side_effect_free(
    quel3_registration: Any,
    options: dict[str, Any],
    exception: type[Exception],
    match: str,
) -> None:
    """Invalid custom targets should leave hardware, cache, and registry unchanged."""
    env = quel3_registration
    registry = env.system.gen_targets
    cache = env.cache.snapshot()
    with pytest.raises(exception, match=match):
        _register_quel3(env, update_backend_settings=True, **options)
    assert env.hardware.calls == []
    assert env.system.gen_targets == registry
    assert env.cache.snapshot() == cache


@pytest.mark.parametrize("update", [False, True])
def test_quel3_rejects_registered_alias_port_move(
    quel3_registration: Any, update: bool
) -> None:
    """A registered alias should not move to another port regardless of its original update flag."""
    env = quel3_registration
    _register_quel3(env, update_backend_settings=update)
    original = env.system.get_target("custom")
    env.context.control_system.port = env.box.get_port(4)
    with pytest.raises(ValueError, match="port"):
        _register_quel3(env, port_number=4, update_backend_settings=True)
    assert env.system.get_target("custom") is original
    assert len(env.hardware.calls) == 1


@pytest.mark.parametrize("update", [None, False, True])
def test_quel3_rejects_cached_alias_port_move(
    quel3_registration: Any,
    update: bool | None,
) -> None:
    """A cached hardware alias on another port should be rejected before deployment."""
    env = quel3_registration
    previous_registry = env.system.gen_targets
    previous_cache = env.cache.snapshot()
    with pytest.raises(ValueError, match="port"):
        _register_quel3(
            env,
            label="elsewhere",
            **({} if update is None else {"update_backend_settings": update}),
        )
    assert env.hardware.calls == []
    assert env.system.gen_targets == previous_registry
    assert env.cache.snapshot() == previous_cache


@pytest.mark.parametrize(
    "problem", ["unwired", "wrong_mux", "missing_input", "different_unit"]
)
def test_quel3_read_registration_rejects_invalid_pair(
    quel3_registration: Any, problem: str
) -> None:
    """Custom READ targets should require their qubit's mux and a same-unit input pair."""
    env = quel3_registration
    env.context.control_system.port = env.box.get_port(1)
    if problem == "unwired":
        env.system.get_mux_by_readout_port = lambda _: None
    elif problem == "wrong_mux":
        env.system.get_mux_by_readout_port = lambda _: SimpleNamespace(index=1)
    elif problem == "missing_input":
        env.system.wiring_info.read_in = []
    else:
        env.system.wiring_info.read_in = [
            (
                env.system.get_mux_by_qubit("Q00"),
                _make_cap_port(box_id="OTHER", port_number=0),
            )
        ]
    with pytest.raises(ValueError, match="Readout port"):
        _register_quel3(
            env,
            port_number=1,
            target_type=TargetType.READ,
            update_backend_settings=True,
        )
    assert env.hardware.calls == []
    assert "custom" not in env.system.gen_targets


def test_quel1_still_requires_explicit_channel() -> None:
    """QuEL-1 should reject omitted channels before defining or registering a target."""
    context, system, backend, _ = _make_context()
    with pytest.raises(TypeError, match="channel_number"):
        context.register_custom_target(
            label="custom", frequency=4.2, box_id="B0", port_number=2, qubit_label="Q00"
        )
    assert backend.define_calls == []
    assert system.added_targets == []


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf"), 1.25])
def test_quel3_registration_validates_margin_before_deployment(
    quel3_registration: Any, margin: float
) -> None:
    """Invalid instrument frequency margins should fail before deployment or target publication."""
    env = quel3_registration
    env.system.control_params.get_frequency_margin = lambda _: margin
    with pytest.raises(ValueError, match="frequency_margin"):
        _register_quel3(env)
    assert "custom" not in env.system.gen_targets
    assert env.hardware.calls == []


def test_quel3_explicit_channel_is_a_logical_reference(quel3_registration: Any) -> None:
    """An explicit valid channel should select the model reference without changing the instrument port."""
    env = quel3_registration
    env.context.control_system.port = env.box.get_port(4)
    _register_quel3(env, port_number=4, channel_number=1, update_backend_settings=True)
    assert env.system.get_target("custom").channel is env.box.get_port(4).channels[1]
    assert env.hardware.calls[0].port_id == "Q3:tx_p04"


def test_registered_custom_targets_compile_and_resolve_from_cache(
    quel3_registration: Any,
) -> None:
    """Custom drive and READ labels should compile into timelines resolved through the deployed cache."""
    from qubex.backend.quel3.execution.resources import Quel3PayloadAnalyzer
    from qubex.measurement.adapters import Quel3MeasurementBackendAdapter
    from qubex.measurement.models import MeasurementConfig, MeasurementSchedule
    from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
    from qubex.pulse import PulseSchedule, Rect

    env = quel3_registration
    _register_quel3(env, update_backend_settings=True)
    env.context.control_system.port = env.box.get_port(1)
    _register_quel3(
        env,
        label="custom-read",
        port_number=1,
        target_type=TargetType.READ,
        update_backend_settings=True,
    )
    with PulseSchedule(["custom", "custom-read"]) as pulses:
        pulses.add("custom", Rect(duration=4, amplitude=0.1, sampling_period=0.4))
        pulses.add("custom-read", Rect(duration=4, amplitude=0.1, sampling_period=0.4))
    schedule = MeasurementSchedule(
        pulse_schedule=pulses,
        capture_schedule=CaptureSchedule(
            captures=[Capture(channels=["custom-read"], start_time=0, duration=4)]
        ),
    )
    adapter = Quel3MeasurementBackendAdapter(
        backend_controller=env.controller, experiment_system=env.system
    )
    request = adapter.build_execution_request(
        schedule=schedule,
        config=MeasurementConfig(
            n_shots=1,
            shot_interval=100,
            shot_averaging=True,
            time_integration=True,
            state_classification=False,
        ),
    )
    payload = cast(Any, request.payload)
    assert set(payload.fixed_timelines) == {"custom", "custom-read"}
    assert payload.fixed_timelines["custom"].frequency_hz == 4.2e9
    assert len(payload.fixed_timelines["custom-read"].capture_windows) == 1
    analyzer = Quel3PayloadAnalyzer(default_sampling_period_ns=0.4)
    analysis = analyzer.analyze_all((payload,), env.cache)
    assert set(analysis[0].payload.fixed_timelines) == {"custom", "custom-read"}
    env.hardware.failure = "read"
    with pytest.raises(RuntimeError, match="read failed"):
        _register_quel3(
            env,
            label="custom-read",
            port_number=1,
            target_type=TargetType.READ,
            update_backend_settings=True,
        )
    with pytest.raises(ValueError, match=r"custom-read.*not cached"):
        analyzer.analyze_all((payload,), env.cache)


def test_custom_target_delegates_to_injected_synchronizer() -> None:
    """Custom registration should use the injected synchronizer without backend type checks."""
    context, system, backend, stub = _make_context()
    events: list[str] = []
    channel = context.control_system.port.channels[0]

    class Synchronizer:
        def resolve_custom_target_channel(self, **kwargs: Any) -> Any:
            assert kwargs["experiment_system"] is system
            assert kwargs["box_id"] == "B0"
            events.append("resolve")
            return channel

        def register_custom_target(self, **kwargs: Any) -> None:
            assert kwargs["experiment_system"] is system
            assert kwargs["update_backend_settings"] is True
            events.append("register")
            system.add_target(kwargs["target"])
            kwargs["push_boxes"](["B0"])

    stub.manager._system_synchronizer = cast(Any, Synchronizer())  # noqa: SLF001
    context.register_custom_target(
        label="custom",
        frequency=4.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        qubit_label="Q00",
        update_backend_settings=True,
    )
    assert events == ["resolve", "register"]
    assert system.get_target("custom").channel is channel
    assert stub.push_calls == [["B0"]]
    assert backend.define_calls == []


def test_quel1_registration_retains_confirmed_push_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canceling QuEL-1 push should retain the registered target and restore the saved cache."""
    context, system, backend, stub = _make_context()
    manager = stub.manager
    events: list[str] = []
    monkeypatch.setattr(manager, "push", SystemManager.push.__get__(manager))
    monkeypatch.setattr(
        system, "get_box", lambda _: SimpleNamespace(id="B0", name="Box"), raising=False
    )
    monkeypatch.setattr(manager, "_supports_box_settings_cache_sync", lambda: True)
    monkeypatch.setattr(
        manager,
        "_sync_backend_settings_to_backend_controller",
        lambda: events.append("restore"),
    )
    monkeypatch.setattr(manager, "_update_cached_state", lambda: events.append("cache"))
    monkeypatch.setattr(
        manager,
        "_sync_experiment_system_to_hardware",
        lambda **_: events.append("hardware"),
    )

    def decline(_: str) -> bool:
        assert len(backend.define_calls) == 1
        assert system.get_target("custom").channel.fnco_freq != 0
        events.append("confirm")
        return False

    monkeypatch.setattr("qubex.system.system_manager.Confirm.ask", decline)
    context.register_custom_target(
        label="custom",
        frequency=4.2,
        box_id="B0",
        port_number=2,
        channel_number=0,
        qubit_label="Q00",
        update_backend_settings=True,
    )
    assert events == ["confirm", "restore", "cache"]
    assert system.get_target("custom").frequency == 4.2
