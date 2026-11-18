"""Tests for instrument addition and replacement through QuEL-3 APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3 import (
    InstrumentConfiguration,
    InstrumentSpec,
    Quel3BackendController,
)
from qubex.backend.quel3.infra import Quel3ResourceReader
from qubex.backend.quel3.infra.quelware_imports import QuelwareInstrumentEntities
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
)
from qubex.backend.quel3.managers.session_workarounds import QuelwareSessionError
from qubex.system.quel3.quel3_system_synchronizer import Quel3SystemSynchronizer
from qubex.system.system_manager import SystemManager


def _spec(alias: str = "Q02", port_id: str = "unit-a:tx_p01") -> InstrumentSpec:
    return InstrumentSpec(
        port_id=port_id,
        alias=alias,
        role="TRANSMITTER",
        frequency_range_min_hz=4e9,
        frequency_range_max_hz=6e9,
    )


def test_modify_target_frequencies_preserves_instrument_width_and_other_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frequency changes should append matching aliases with their existing range widths."""
    manager = Quel3ConfigurationManager()
    controller = Quel3BackendController(configuration_manager=manager)
    original = _spec("Q00")
    sibling = _spec("Q01")
    monkeypatch.setattr(
        controller,
        "get_instrument_configuration",
        lambda: InstrumentConfiguration(instruments=(original, sibling)),
    )
    deployed: list[InstrumentSpec] = []

    def deploy(*, instrument: InstrumentSpec, **kwargs: Any) -> Any:
        deployed.append(instrument)
        return None

    monkeypatch.setattr(manager, "replace_instruments", deploy)
    controller.modify_target_frequencies({"Q00": 7.0})

    assert deployed == [
        InstrumentSpec(
            alias="Q00",
            port_id=original.port_id,
            role=original.role,
            frequency_range_min_hz=6e9,
            frequency_range_max_hz=8e9,
        )
    ]


def _info(
    alias: str,
    port_id: str = "unit-a:tx_p01",
    *,
    resource_id: str | None = None,
    frequency_range_min_hz: float = 4e9,
    frequency_range_max_hz: float = 6e9,
) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id or f"{port_id}:{alias}",
            port_id=port_id,
            definition=SimpleNamespace(
                alias=f"{port_id.partition(':')[0]}:{alias}",
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=frequency_range_min_hz,
                    frequency_range_max=frequency_range_max_hz,
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000),
        ),
    )


class _Hardware:
    def __init__(self) -> None:
        self.infos = [_info("Q00"), _info("Q01", "unit-b:tx_p02")]
        self.deploy_calls: list[tuple[str, bool]] = []
        self.read_calls: list[tuple[str, ...]] = []
        self.id_read_calls: list[tuple[str, ...]] = []
        self.closed_sessions = 0
        self.fail_after_write = False
        self.fail_deployment_number: int | None = None

    async def __aenter__(self) -> _Hardware:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def create_session(self, resource_ids: Sequence[str], **kwargs: object) -> object:
        hardware = self

        class _Session:
            token = "single-deploy-session"  # noqa: S105

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *_: object) -> None:
                hardware.closed_sessions += 1

            async def deploy_instruments(
                self,
                port_id: str,
                *,
                definitions: list[Any],
                append: bool,
            ) -> list[InstrumentInfoProtocol]:
                hardware.deploy_calls.append((port_id, append))
                if not append:
                    hardware.infos = [
                        info for info in hardware.infos if info.port_id != port_id
                    ]
                else:
                    aliases = {definition.alias for definition in definitions}
                    hardware.infos = [
                        info
                        for info in hardware.infos
                        if info.port_id != port_id
                        or InstrumentCache.alias_for(info) not in aliases
                    ]
                hardware.infos.extend(
                    _info(
                        definition.alias,
                        port_id,
                        resource_id=f"{port_id}:{definition.alias}:deployment-{len(hardware.deploy_calls)}",
                        frequency_range_min_hz=definition.profile.frequency_range_min,
                        frequency_range_max_hz=definition.profile.frequency_range_max,
                    )
                    for definition in definitions
                )
                if hardware.fail_after_write or hardware.fail_deployment_number == len(
                    hardware.deploy_calls
                ):
                    raise RuntimeError("response lost after deployment")
                return [info for info in hardware.infos if info.port_id == port_id]

        return _Session()

    def read_instrument_infos(
        self,
        *,
        port_ids: Sequence[str] = (),
        parallel: bool = True,
        unit_labels: Sequence[str] = (),
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.read_calls.append(tuple(port_ids))
        return tuple(
            info for info in self.infos if not port_ids or info.port_id in port_ids
        )

    def read_instrument_infos_by_ids(
        self, instrument_ids: Sequence[str], *, parallel: bool = True
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.id_read_calls.append(tuple(instrument_ids))
        return tuple(info for info in self.infos if info.id in instrument_ids)


@pytest.fixture
def deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache]:
    """Provide a real manager and cache backed by in-memory hardware IO."""
    hardware = _Hardware()
    manager = Quel3ConfigurationManager()
    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: lambda *args: hardware
    )
    entities = QuelwareInstrumentEntities(
        fixed_timeline_profile_factory=cast(Any, SimpleNamespace),
        instrument_definition_factory=cast(Any, SimpleNamespace),
        instrument_mode_namespace=cast(
            Any, SimpleNamespace(FIXED_TIMELINE="FIXED_TIMELINE")
        ),
        instrument_role_namespace=cast(Any, SimpleNamespace(TRANSMITTER="TRANSMITTER")),
    )
    monkeypatch.setattr(manager, "_load_instrument_entities", lambda: entities)
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=hardware.infos)
    return manager, hardware, cache


@pytest.mark.parametrize("parallel", [False, True])
def test_deployment_request_returns_only_requested_instruments(deployment, parallel):
    """Each port should return its requested instruments even when append reports siblings."""
    manager, hardware, _ = deployment
    other_port = "unit-a:tx_p02"
    hardware.infos.append(_info("another-user", other_port))
    specifications = (_spec("Q02"), _spec("Q03", other_port))

    infos = asyncio.run(
        manager._request_instrument_deployment(  # noqa: SLF001
            specifications=specifications, append=True, parallel=parallel
        )
    )

    assert [(InstrumentCache.alias_for(info), info.port_id) for info in infos] == [
        (spec.alias, spec.port_id) for spec in specifications
    ]


@pytest.mark.parametrize(
    ("failure", "append", "message"),
    [
        ("missing", True, "Missing"),
        ("duplicate_alias", True, "Duplicate alias"),
        ("duplicate_id", True, "Duplicate resource ID"),
        ("swapped_ports", True, "unexpected port"),
        ("extra", False, "Unexpected alias"),
        ("empty_alias", False, "alias must not be empty"),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_port_response_is_not_retried(
    deployment, monkeypatch, failure, append, message, parallel
):
    """An invalid response should identify the port and propagate without retrying the write."""
    manager, hardware, _ = deployment
    ports = ("unit-a:tx_p01", "unit-a:tx_p02")
    specifications = (_spec("A", ports[0]), _spec("B", ports[1]))
    session = hardware.create_session(ports)
    calls = []

    async def deploy(port_id, **kwargs):
        calls.append(port_id)
        spec = specifications[ports.index(port_id)]
        info = _info(spec.alias, port_id)
        if failure == "empty_alias":
            return [_info("", port_id)]
        if failure == "missing":
            return []
        if failure == "duplicate_alias":
            return [info, _info(spec.alias, port_id, resource_id="another-id")]
        if failure == "duplicate_id":
            return [info, _info("sibling", port_id, resource_id=info.id)]
        if failure == "swapped_ports":
            other = specifications[1 - ports.index(port_id)]
            return [_info(other.alias, other.port_id)]
        return [info, _info("unexpected", port_id)]

    monkeypatch.setattr(session, "deploy_instruments", deploy)
    monkeypatch.setattr(hardware, "create_session", lambda *args, **kwargs: session)
    with pytest.raises(ValueError, match=message) as error:
        asyncio.run(
            manager._request_instrument_deployment(  # noqa: SLF001
                specifications=specifications, append=append, parallel=parallel
            )
        )
    assert ports[0] in str(error.value)
    assert calls == list(ports)
    assert hardware.closed_sessions == 1


@pytest.mark.parametrize("append", [None, False])
def test_batch_deployment_preserves_siblings_unless_replacement_is_explicit(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    append: bool | None,
) -> None:
    """Batch deployment should preserve siblings by default and replace only selected ports on request."""
    manager, hardware, _ = deployment
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    untouched = hardware.infos[1]
    controller = Quel3BackendController(
        configuration_manager=manager,
        resource_reader=cast(Quel3ResourceReader, hardware),
    )
    controller.refresh_instrument_cache()
    hardware.read_calls.clear()
    # An existing alias may be discovered in an append response for the first time.
    hardware.infos.append(_info("uncached"))
    original = controller.get_instrument_configuration()
    configuration = InstrumentConfiguration(
        instruments=(
            _spec("Q00").model_copy(update={"frequency_range_min_hz": 4.2e9}),
            _spec("Q02"),
        )
    )

    if append is None:
        result = controller.deploy_instruments(configuration=configuration)
    else:
        result = controller.deploy_instruments(
            configuration=configuration, append=append
        )

    expected = {"Q00", "Q01", "Q02"}
    if append is None:
        expected.add("Q03")
        assert sibling in hardware.infos
    else:
        assert sibling not in hardware.infos
    assert set(result) == {"Q00", "Q02"}
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []
    assert result["Q00"].id.endswith("deployment-1")
    profile = result["Q00"].definition.profile
    assert profile is not None
    assert profile.frequency_range_min == 4.2e9
    assert untouched in hardware.infos
    cached = controller.get_instrument_configuration()
    assert {spec.alias for spec in cached.instruments} == expected
    assert next(spec for spec in cached.instruments if spec.alias == "Q01") == next(
        spec for spec in original.instruments if spec.alias == "Q01"
    )
    assert hardware.deploy_calls == [("unit-a:tx_p01", append is None)]


@pytest.mark.parametrize("append", [True, False])
def test_batch_alias_conflict_fails_before_writes_or_cache_invalidation(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    append: bool,
) -> None:
    """A batch alias collision with an untouched port should preserve hardware and cache."""
    manager, hardware, cache = deployment
    before = cache.snapshot()

    with pytest.raises(ValueError, match="already cached on port"):
        manager.deploy_instruments(
            configuration=InstrumentConfiguration(instruments=(_spec("Q01"),)),
            instrument_cache=cache,
            append=append,
        )

    assert hardware.deploy_calls == []
    assert hardware.read_calls == []
    assert cache.snapshot() == before


@pytest.fixture
def frequency_controller(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> tuple[Quel3BackendController, _Hardware]:
    """Provide a controller with cached instruments and recorded hardware requests."""
    manager, hardware, _ = deployment
    controller = Quel3BackendController(
        configuration_manager=manager,
        resource_reader=cast(Quel3ResourceReader, hardware),
    )
    controller.refresh_instrument_cache()
    hardware.read_calls.clear()
    return controller, hardware


@pytest.mark.parametrize("frequency", [3.999999, 4.099999, 5.900001, 6.000001, 7.0])
def test_controller_frequency_change_caches_deployed_target(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
    frequency: float,
) -> None:
    """Insufficient frequency margins should redeploy only the target with its existing width."""
    controller, hardware = frequency_controller
    controller.modify_target_frequencies({"Q00": frequency})

    assert hardware.id_read_calls == []
    assert hardware.read_calls == []
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    specs = {
        spec.alias: spec
        for spec in controller.get_instrument_configuration().instruments
    }
    assert set(specs) == {"Q00", "Q01"}
    assert specs["Q00"].frequency_range_min_hz == pytest.approx(
        frequency * 1e9 - 1e9, rel=0, abs=1e-6
    )
    assert specs["Q00"].frequency_range_max_hz == pytest.approx(
        frequency * 1e9 + 1e9, rel=0, abs=1e-6
    )


@pytest.mark.parametrize("frequency", [4.1, 4.100001, 5.0, 5.899999, 5.9])
def test_controller_reuses_instrument_for_in_range_frequency(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
    frequency: float,
) -> None:
    """Frequencies with at least the default margin should reuse instruments without hardware IO."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration()
    original_infos = tuple(hardware.infos)
    original_hash = controller.hash

    controller.modify_target_frequencies({"Q00": frequency})

    assert hardware.deploy_calls == []
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []
    assert controller.get_instrument_configuration() == original
    assert controller.hash == original_hash
    assert all(
        current is previous
        for current, previous in zip(hardware.infos, original_infos, strict=True)
    )


def test_controller_only_redeploys_out_of_range_targets(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
) -> None:
    """Mixed frequency changes should redeploy only out-of-range targets."""
    controller, hardware = frequency_controller
    original = hardware.infos[0]

    controller.modify_target_frequencies({"Q00": 5.5, "Q01": 7.0})

    assert hardware.deploy_calls == [("unit-b:tx_p02", True)]
    assert hardware.id_read_calls == []
    assert hardware.read_calls == []
    assert hardware.infos[0] is original


def test_controller_reuses_range_confirmed_after_redeployment(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
) -> None:
    """Later frequency changes should reuse the range returned by redeployment."""
    controller, hardware = frequency_controller

    controller.modify_target_frequencies({"Q00": 7.0})
    updated = controller.get_instrument_configuration()
    controller.modify_target_frequencies({"Q00": 7.5})

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.id_read_calls == []
    assert hardware.read_calls == []
    assert controller.get_instrument_configuration() == updated


@pytest.mark.parametrize(
    ("margin_ghz", "frequency", "redeploy"),
    [
        (None, 4.1, False),
        (None, 4.099999, True),
        (None, 5.9, False),
        (None, 5.900001, True),
        (0.0, 4.0, False),
        (0.0, 6.0, False),
        (0.2, 4.2, False),
        (0.2, 4.199999, True),
        (0.2, 5.8, False),
        (0.2, 5.800001, True),
        (1.1, 5.0, True),
    ],
)
def test_controller_guarantees_requested_frequency_margin(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
    margin_ghz: float | None,
    frequency: float,
    redeploy: bool,
) -> None:
    """The requested half-width should determine reuse and expand narrower deployment ranges."""
    controller, hardware = frequency_controller

    controller.modify_target_frequencies(
        {"Q00": frequency}, min_frequency_margin_ghz=margin_ghz
    )

    assert hardware.deploy_calls == ([("unit-a:tx_p01", True)] if redeploy else [])
    assert hardware.id_read_calls == []
    assert hardware.read_calls == []
    specifications = {
        spec.alias: spec
        for spec in controller.get_instrument_configuration().instruments
    }
    expected_margin_hz = max(1.0, 0.1 if margin_ghz is None else margin_ghz) * 1e9
    expected_lower_hz = frequency * 1e9 - expected_margin_hz if redeploy else 4e9
    expected_upper_hz = frequency * 1e9 + expected_margin_hz if redeploy else 6e9
    assert specifications["Q00"].frequency_range_min_hz == pytest.approx(
        expected_lower_hz, rel=0, abs=1e-6
    )
    assert specifications["Q00"].frequency_range_max_hz == pytest.approx(
        expected_upper_hz, rel=0, abs=1e-6
    )
    assert specifications["Q01"] == _spec("Q01", "unit-b:tx_p02")


@pytest.mark.parametrize(
    "margin_ghz", [-0.1, float("nan"), float("inf"), -float("inf"), 1e308]
)
def test_controller_rejects_invalid_frequency_margin_without_hardware_io(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
    margin_ghz: float,
) -> None:
    """Invalid minimum margins should fail before changing hardware or cached instruments."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration()

    with pytest.raises(ValueError, match="min_frequency_margin_ghz"):
        controller.modify_target_frequencies(
            {"Q00": 7.0}, min_frequency_margin_ghz=margin_ghz
        )

    assert hardware.deploy_calls == []
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []
    assert controller.get_instrument_configuration() == original


def test_controller_expands_narrow_instrument_to_default_frequency_margin(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
) -> None:
    """A narrow cached instrument should expand to the default margin around an unchanged frequency."""
    controller, hardware = frequency_controller
    hardware.infos[0] = _info(
        "Q00", frequency_range_min_hz=4.95e9, frequency_range_max_hz=5.05e9
    )
    controller.refresh_instrument_cache()
    hardware.read_calls.clear()

    controller.modify_target_frequencies({"Q00": 5.0})

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.id_read_calls == []
    specification = controller.get_instrument_configuration().instruments[0]
    assert specification.frequency_range_min_hz == 4.9e9
    assert specification.frequency_range_max_hz == 5.1e9


@pytest.mark.parametrize(("original_frequency", "frequency"), [(5.0, 5.5), (7.0, 5.5)])
@pytest.mark.parametrize("fail_in_context", [False, True])
def test_modified_frequencies_restores_software_without_redeployment(
    frequency_controller: tuple[Quel3BackendController, _Hardware],
    fail_in_context: bool,
    original_frequency: float,
    frequency: float,
) -> None:
    """Temporary carrier changes restore software without changing deployed instruments."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration()
    target = SimpleNamespace(label="Q00", frequency=original_frequency)

    def modify(frequencies: dict[str, float]) -> None:
        target.frequency = frequencies[target.label]

    experiment_system = SimpleNamespace(
        targets=[target], modify_target_frequencies=modify
    )
    # Isolate the context manager from singleton initialization and hardware setup.
    manager = object.__new__(SystemManager)
    manager._experiment_system = cast(Any, experiment_system)  # noqa: SLF001
    manager._backend_controller = controller  # noqa: SLF001
    manager._system_synchronizer = Quel3SystemSynchronizer(  # noqa: SLF001
        backend_controller=controller
    )

    def run_context() -> None:
        with manager.modified_frequencies({"Q00": frequency}):
            assert target.frequency == frequency
            if fail_in_context:
                raise RuntimeError("measurement failed")

    if fail_in_context:
        with pytest.raises(RuntimeError, match="measurement failed"):
            run_context()
    else:
        run_context()

    assert target.frequency == original_frequency
    assert controller.get_instrument_configuration() == original
    assert hardware.deploy_calls == []
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []


def _frequency_system(
    controller: Quel3BackendController, frequencies: dict[str, float]
) -> SystemManager:
    """Provide an isolated system sharing the controller's instrument cache."""
    targets = {
        label: SimpleNamespace(label=label, frequency=frequency)
        for label, frequency in frequencies.items()
    }

    def modify(values: dict[str, float]) -> None:
        for label, frequency in values.items():
            targets[label].frequency = frequency

    system = object.__new__(SystemManager)
    system._experiment_system = cast(  # noqa: SLF001
        Any,
        SimpleNamespace(
            targets=list(targets.values()),
            get_target=targets.__getitem__,
            modify_target_frequencies=modify,
        ),
    )
    system._backend_controller = controller  # noqa: SLF001
    system._system_synchronizer = Quel3SystemSynchronizer(backend_controller=controller)  # noqa: SLF001
    return system


@pytest.mark.parametrize("fail_acquisition", [False, True])
def test_frequency_override_restoration_uses_deployment_response(
    frequency_controller, monkeypatch, fail_acquisition
):
    """Restoration should succeed without reading another MUX's instruments."""
    controller, hardware = frequency_controller
    hardware.infos[1] = _info("Q01", "unit-a:tx_p02")
    controller.refresh_instrument_cache()
    original = controller.get_instrument_configuration()
    system = _frequency_system(controller, {"Q00": 5.0})

    def fail_broad_read(**kwargs):
        raise RuntimeError("another MUX's instrument was removed")

    monkeypatch.setattr(hardware, "read_instrument_infos", fail_broad_read)

    def acquire():
        with system.modified_frequencies({"Q00": 7.0}):
            if fail_acquisition:
                raise RuntimeError("acquisition failed")

    if fail_acquisition:
        with pytest.raises(RuntimeError, match="acquisition failed"):
            acquire()
    else:
        acquire()

    assert controller.get_instrument_configuration() == original
    assert hardware.id_read_calls == []
    assert system.experiment_system.get_target("Q00").frequency == 5.0


@pytest.mark.parametrize("fail_acquisition", [False, True])
def test_frequency_override_restores_exact_definition_and_keeps_siblings(
    frequency_controller, fail_acquisition
):
    """Out-of-range overrides redeploy and restore only their changed instruments."""
    controller, hardware = frequency_controller
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    controller.refresh_instrument_cache()
    hardware.read_calls.clear()
    original = controller.get_instrument_configuration()
    system = _frequency_system(controller, {"Q00": 5.8, "Q01": 5.0})

    def acquire():
        with system.modified_frequencies({"Q00": 7.0, "Q01": 5.5}):
            assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
            if fail_acquisition:
                raise RuntimeError("acquisition failed")

    if fail_acquisition:
        with pytest.raises(RuntimeError, match="acquisition failed"):
            acquire()
    else:
        acquire()

    assert controller.get_instrument_configuration() == original
    assert system.experiment_system.get_target("Q00").frequency == 5.8
    assert system.experiment_system.get_target("Q01").frequency == 5.0
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)] * 2
    assert (
        next(
            info for info in hardware.infos if InstrumentCache.alias_for(info) == "Q03"
        )
        is sibling
    )


@pytest.mark.parametrize("inner_frequency", [7.5, 10.0])
def test_frequency_override_restores_nested_entry_definitions(
    frequency_controller, inner_frequency
):
    """Nested overrides restore their own definitions without redeploying unchanged ranges."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration()
    system = _frequency_system(controller, {"Q00": 5.8})
    with system.modified_frequencies({"Q00": 7.0}):
        outer = controller.get_instrument_configuration()
        with system.modified_frequencies({"Q00": inner_frequency}):
            assert (
                system.experiment_system.get_target("Q00").frequency == inner_frequency
            )
        assert controller.get_instrument_configuration() == outer
        assert system.experiment_system.get_target("Q00").frequency == 7.0
    assert controller.get_instrument_configuration() == original
    assert system.experiment_system.get_target("Q00").frequency == 5.8
    assert len(hardware.deploy_calls) == (2 if inner_frequency == 7.5 else 4)


@pytest.mark.parametrize("failed_deployment", [1, 2])
def test_frequency_override_restores_after_partial_entry_failure(
    frequency_controller, failed_deployment
):
    """Failed deployment restores attempted instruments before propagating the entry error."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration()
    system = _frequency_system(controller, {"Q00": 5.8, "Q01": 5.0})
    hardware.fail_deployment_number = failed_deployment
    with (
        pytest.raises(QuelwareSessionError, match="without retry"),
        system.modified_frequencies({"Q00": 7.0, "Q01": 7.0}),
    ):
        pytest.fail("Entry must fail")
    assert controller.get_instrument_configuration() == original
    assert system.experiment_system.get_target("Q00").frequency == 5.8
    assert len(hardware.deploy_calls) == failed_deployment * 2


def test_frequency_override_invalidates_failed_restoration(frequency_controller):
    """Failed restoration propagates while restoring model frequencies and invalidating the cache."""
    controller, hardware = frequency_controller
    system = _frequency_system(controller, {"Q00": 5.8})
    hardware.fail_deployment_number = 2
    with (
        pytest.raises(QuelwareSessionError, match="without retry"),
        system.modified_frequencies({"Q00": 7.0}),
    ):
        assert system.experiment_system.get_target("Q00").frequency == 7.0
    assert system.experiment_system.get_target("Q00").frequency == 5.8
    assert {s.alias for s in controller.get_instrument_configuration().instruments} == {
        "Q01"
    }


def test_frequency_override_attempts_remaining_restorations_after_failure(
    frequency_controller,
):
    """One failed restoration does not prevent restoring the other changed instruments."""
    controller, hardware = frequency_controller
    original = controller.get_instrument_configuration().instruments[0]
    system = _frequency_system(controller, {"Q00": 5.8, "Q01": 5.0})
    hardware.fail_deployment_number = 3
    with (
        pytest.raises(QuelwareSessionError, match="without retry"),
        system.modified_frequencies({"Q00": 7.0, "Q01": 7.0}),
    ):
        assert system.experiment_system.get_target("Q00").frequency == 7.0
    assert controller.get_instrument_configuration().instruments == (original,)
    assert system.experiment_system.get_target("Q00").frequency == 5.8
    assert system.experiment_system.get_target("Q01").frequency == 5.0
    assert len(hardware.deploy_calls) == 4


@pytest.mark.parametrize("fail_acquisition", [False, True])
def test_sweep_frequency_overrides_do_not_redeploy_or_read_back(
    frequency_controller, fail_acquisition
):
    """Planned points and references preserve instruments even when restoring an outside carrier."""
    controller, hardware = frequency_controller
    system = _frequency_system(controller, {"Q00": 5.0})
    original = controller.get_instrument_configuration()
    plan = controller.plan_frequency_sweep(
        [6.2, 7.8, 8.0, 9.4], target="Q00", overlap=True
    )
    assert [segment.indices for segment in plan] == [(0, 1), (2, 3)]

    def acquire(segment):
        with controller.modified_sweep_configuration(segment.configuration):
            entry = controller.get_instrument_configuration()
            entry_hash = controller.hash
            counts = (
                len(hardware.deploy_calls),
                len(hardware.read_calls),
                len(hardware.id_read_calls),
            )
            points = segment.values
            if segment.reference_value is not None:
                points = (segment.reference_value, *points)
            for frequency in points:
                with system.modified_frequencies({"Q00": frequency}):
                    assert controller.get_instrument_configuration() == entry
                    if fail_acquisition:
                        raise RuntimeError("acquisition failed")
                assert system.experiment_system.get_target("Q00").frequency == 5.0
                assert controller.hash == entry_hash
                assert counts == (
                    len(hardware.deploy_calls),
                    len(hardware.read_calls),
                    len(hardware.id_read_calls),
                )

    for number, segment in enumerate(plan, start=1):
        if fail_acquisition:
            with pytest.raises(RuntimeError, match="acquisition failed"):
                acquire(segment)
        else:
            acquire(segment)
        assert controller.get_instrument_configuration() == original
        assert system.experiment_system.get_target("Q00").frequency == 5.0
        assert len(hardware.deploy_calls) == number * 2


@pytest.mark.parametrize("append", [True, False])
def test_single_deploy_returns_cached_response(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    append: bool,
) -> None:
    """Single deployment should honor append and preserve other ports in the cache."""
    manager, hardware, cache = deployment
    untouched = cache.get("Q01")

    result = manager.deploy_instrument(
        instrument=_spec(),
        append=append,
        instrument_cache=cache,
        parallel=False,
    )

    assert hardware.deploy_calls == [("unit-a:tx_p01", append)]
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []
    assert cache.get("Q02") is result
    assert cache.get("Q01") is untouched
    assert set(cache.snapshot()) == (
        {"Q00", "Q01", "Q02"} if append else {"Q01", "Q02"}
    )


def test_controller_single_deploy_defaults_to_append(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> None:
    """The controller should add one instrument by default and expose the updated configuration."""
    manager, hardware, _ = deployment
    controller = Quel3BackendController(
        configuration_manager=manager,
        resource_reader=cast(Quel3ResourceReader, hardware),
    )
    controller.refresh_instrument_cache()

    result = controller.deploy_instrument(instrument=_spec())

    assert result.id == "unit-a:tx_p01:Q02:deployment-1"
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert {
        spec.alias for spec in controller.get_instrument_configuration().instruments
    } == {"Q00", "Q01", "Q02"}


@pytest.mark.parametrize("cache_populated", [True, False])
def test_append_replaces_existing_alias_without_a_preflight_read(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    cache_populated: bool,
) -> None:
    """Append should cache the replaced alias and retain already cached siblings."""
    manager, hardware, cache = deployment
    old = hardware.infos[0]
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    cache.replace_all(instrument_infos=hardware.infos if cache_populated else ())

    result = manager.deploy_instrument(
        instrument=_spec(alias="Q00"),
        instrument_cache=cache,
    )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.read_calls == []
    assert hardware.id_read_calls == []
    assert result.id != old.id
    assert cache.get("Q00") is result
    assert sibling in hardware.infos
    if cache_populated:
        assert cache.get("Q03") is sibling
    assert set(cache.snapshot()) == (
        {"Q00", "Q01", "Q03"} if cache_populated else {"Q00"}
    )


def test_append_does_not_retry_after_an_uncertain_write(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> None:
    """An uncertain append should run once and invalidate only the deployed alias."""
    manager, hardware, cache = deployment
    hardware.fail_after_write = True
    untouched = cache.get("Q01")

    with pytest.raises(QuelwareSessionError, match="without retry"):
        manager.deploy_instrument(
            instrument=_spec("Q00"),
            instrument_cache=cache,
        )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.closed_sessions == 1
    assert cache.snapshot() == {"Q01": untouched}


def test_single_deploy_rejects_alias_on_another_cached_port(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> None:
    """Cross-port alias collisions should fail before touching hardware or the cache."""
    manager, hardware, cache = deployment
    before = cache.snapshot()
    with pytest.raises(ValueError, match="port"):
        manager.deploy_instrument(
            instrument=_spec("Q01"),
            instrument_cache=cache,
        )
    assert hardware.deploy_calls == []
    assert cache.snapshot() == before
