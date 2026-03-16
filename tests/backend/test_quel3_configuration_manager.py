"""Tests for QuEL-3 backend configuration manager behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    configuration_manager as configuration_manager_module,
)
from qubex.backend.quel3.models import InstrumentDeployRequest


@dataclass(frozen=True)
class _CachedProfile:
    frequency_range_min: float
    frequency_range_max: float


@dataclass(frozen=True)
class _CachedRole:
    name: str


@dataclass(frozen=True)
class _CachedDefinition:
    alias: str
    role: object | None = None
    profile: object | None = None


@dataclass(frozen=True)
class _CachedInstrumentInfo:
    id: str
    port_id: str
    definition: _CachedDefinition


def test_deploy_instruments_calls_session_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given deploy requests, backend configuration manager should call session deploy."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(
            self, *, alias: str, mode: object, role: object, profile: _Profile
        ):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"
        TRANSCEIVER = "transceiver"

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    deploy_calls: list[tuple[str, list[_Definition]]] = []
    create_session_calls: list[tuple[str, ...]] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            assert append is True
            deploy_calls.append((port_id, definitions))
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}",
                    port_id=port_id,
                    definition=definitions[0],
                )
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return _FakeSession()

    fake_client = _FakeClient()
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: fake_client,
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert create_session_calls == [("quel3-02-a01:tx_p02",)]
    assert len(deploy_calls) == 1
    port_id, definitions = deploy_calls[0]
    assert port_id == "quel3-02-a01:tx_p02"
    definition = definitions[0]
    assert definition.mode == "fixed_timeline"
    assert definition.role == "transmitter"
    assert definition.profile.frequency_range_min == pytest.approx(4.1e9)
    assert definition.profile.frequency_range_max == pytest.approx(4.3e9)
    assert definition.alias == "Q00"
    assert manager.target_alias_map == {"Q00": definition.alias}
    assert definition.alias in deployed


def test_deploy_instruments_clears_cache_for_empty_requests() -> None:
    """Given empty requests, backend configuration manager should clear deployment cache."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
    )
    manager._last_deployed_instrument_infos = {  # noqa: SLF001
        request.alias: (
            _CachedInstrumentInfo(
                id="inst-q00",
                port_id=request.port_id,
                definition=_CachedDefinition(alias=request.alias),
            ),
        )
    }
    manager._target_alias_map = {"Q00": request.alias}  # noqa: SLF001

    deployed = manager.deploy_instruments(requests=())

    assert deployed == {}
    assert manager.last_deployed_instrument_infos == {}
    assert manager.target_alias_map == {}


def test_deploy_instruments_groups_requests_by_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given same-port requests, backend configuration manager should batch one deploy call."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    deploy_calls: list[tuple[str, list[_Definition], bool]] = []
    create_session_calls: list[tuple[str, ...]] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            deploy_calls.append((port_id, definitions, append))
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}:{index}",
                    port_id=port_id,
                    definition=definition,
                )
                for index, definition in enumerate(definitions)
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return _FakeSession()

    fake_client = _FakeClient()
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: fake_client,
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
        ),
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q00-CR",
            target_labels=("Q00-CR",),
        ),
    )

    deployed = manager.deploy_instruments(requests=requests)

    assert create_session_calls == [("quel3-02-a01:tx_p04",)]
    assert len(deploy_calls) == 1
    assert deploy_calls[0][0] == "quel3-02-a01:tx_p04"
    assert deploy_calls[0][2] is True
    assert [definition.alias for definition in deploy_calls[0][1]] == [
        "Q00",
        "Q00-CR",
    ]
    assert manager.target_alias_map == {
        "Q00": "Q00",
        "Q00-CR": "Q00-CR",
    }
    assert set(deployed) == {"Q00", "Q00-CR"}
    assert deployed["Q00"][0].id == "id:quel3-02-a01:tx_p04:0"
    assert deployed["Q00-CR"][0].id == "id:quel3-02-a01:tx_p04:1"


def test_deploy_instruments_uses_one_session_for_all_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given multiple ports, backend configuration manager should reuse one session."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    create_session_calls: list[tuple[str, ...]] = []
    deploy_calls: list[str] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            assert append is True
            deploy_calls.append(port_id)
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}:{definition.alias}",
                    port_id=port_id,
                    definition=definition,
                )
                for definition in definitions
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return _FakeSession()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
        ),
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p06",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q01",
            target_labels=("Q01",),
        ),
    )

    deployed = manager.deploy_instruments(requests=requests)

    assert create_session_calls == [
        ("quel3-02-a01:tx_p04", "quel3-02-a01:tx_p06"),
    ]
    assert deploy_calls == ["quel3-02-a01:tx_p04", "quel3-02-a01:tx_p06"]
    assert set(deployed) == {"Q00", "Q01"}


def test_deploy_instruments_parallelizes_ports_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given multiple ports, deploy_instruments should run port batches concurrently."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    class _Probe:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

    probe = _Probe()

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            assert append is True
            probe.active += 1
            probe.max_active = max(probe.max_active, probe.active)
            await asyncio.sleep(0)
            probe.active -= 1
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}:{definition.alias}",
                    port_id=port_id,
                    definition=definition,
                )
                for definition in definitions
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            del resource_ids
            return _FakeSession()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    manager.deploy_instruments(
        requests=(
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
                target_labels=("Q00",),
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
                target_labels=("Q01",),
            ),
        )
    )

    assert probe.max_active == 2


def test_deploy_instruments_parallel_false_serializes_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel false, deploy_instruments should deploy port batches serially."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    class _Probe:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

    probe = _Probe()

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            assert append is True
            probe.active += 1
            probe.max_active = max(probe.max_active, probe.active)
            await asyncio.sleep(0)
            probe.active -= 1
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}:{definition.alias}",
                    port_id=port_id,
                    definition=definition,
                )
                for definition in definitions
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            del resource_ids
            return _FakeSession()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    manager.deploy_instruments(
        requests=(
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
                target_labels=("Q00",),
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
                target_labels=("Q01",),
            ),
        ),
        parallel=False,
    )

    assert probe.max_active == 1


def test_load_client_factory_uses_configured_client_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given standalone runtime options, client factory loading should use that runtime."""
    captured: dict[str, object] = {}
    fake_client_factory = object()
    monkeypatch.setattr(
        configuration_manager_module,
        "load_quelware_client_factory",
        lambda *, client_mode, standalone_unit_label: (
            captured.update(
                {
                    "client_mode": client_mode,
                    "standalone_unit_label": standalone_unit_label,
                }
            )
            or fake_client_factory
        ),
    )
    manager = Quel3ConfigurationManager(
        quelware_endpoint="worker-host",
        quelware_port=61000,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
    )

    client_factory = manager._load_quelware_client_factory()  # noqa: SLF001

    assert client_factory is fake_client_factory
    assert captured == {
        "client_mode": "standalone",
        "standalone_unit_label": "quel3-02-a01",
    }


def test_refresh_instrument_cache_loads_existing_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given existing quelware instruments, refreshing cache should expose alias mappings."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Category:
        name = "INSTRUMENT"

    class _ResourceInfo:
        def __init__(self, resource_id: str) -> None:
            self.id = resource_id
            self.category = _Category()

    class _Definition:
        def __init__(self, alias: str) -> None:
            self.alias = alias

    class _InstrumentInfo:
        def __init__(self, alias: str, port_id: str) -> None:
            self.definition = _Definition(alias)
            self.port_id = port_id

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def list_resource_infos(self) -> list[object]:
            return [_ResourceInfo("inst-q00"), _ResourceInfo("inst-rq00")]

        async def get_instrument_info(self, resource_id: str) -> object:
            infos = {
                "inst-q00": _InstrumentInfo("Q00", "quel3-02-a01:tx_p04"),
                "inst-rq00": _InstrumentInfo("RQ00", "quel3-02-a01:trx_p00p04"),
            }
            return infos[resource_id]

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )

    cached = manager.refresh_instrument_cache()

    assert set(cached.keys()) == {"Q00", "RQ00"}
    assert manager.target_alias_map == {"Q00": "Q00", "RQ00": "RQ00"}
    assert set(manager.last_deployed_instrument_infos.keys()) == {"Q00", "RQ00"}


def test_fetch_backend_settings_from_hardware_groups_instruments_by_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware instruments, hardware fetch should normalize them per box."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _InstrumentCategory:
        name = "INSTRUMENT"

    class _PortCategory:
        name = "PORT"

    class _ResourceInfo:
        def __init__(self, resource_id: str, category: object) -> None:
            self.id = resource_id
            self.category = category

    class _Role:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Definition:
        def __init__(self, alias: str, role: object) -> None:
            self.alias = alias
            self.role = role

    class _InstrumentInfo:
        def __init__(
            self, resource_id: str, alias: str, port_id: str, role: str
        ) -> None:
            self.id = resource_id
            self.port_id = port_id
            self.definition = _Definition(alias, _Role(role))

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def list_resource_infos(self) -> list[object]:
            return [
                _ResourceInfo("inst-q00", _InstrumentCategory()),
                _ResourceInfo("inst-rq00", _InstrumentCategory()),
                _ResourceInfo("inst-other", _InstrumentCategory()),
                _ResourceInfo("port-q00", _PortCategory()),
            ]

        async def get_instrument_info(self, resource_id: str) -> object:
            infos = {
                "inst-q00": _InstrumentInfo(
                    "inst-q00",
                    "Q00",
                    "quel3-02-a01:tx_p04",
                    "TRANSMITTER",
                ),
                "inst-rq00": _InstrumentInfo(
                    "inst-rq00",
                    "RQ00",
                    "quel3-02-a02:trx_p00p04",
                    "TRANSCEIVER",
                ),
                "inst-other": _InstrumentInfo(
                    "inst-other",
                    "Q99",
                    "quel3-02-a99:tx_p01",
                    "TRANSMITTER",
                ),
            }
            return infos[resource_id]

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )

    fetched = manager.fetch_backend_settings_from_hardware(
        unit_labels_by_box_id={
            "BOX1": "quel3-02-a01",
            "BOX2": "quel3-02-a02",
            "BOX3": "quel3-02-a03",
        },
        parallel=False,
    )

    assert fetched == {
        "BOX1": {
            "instruments": {
                "Q00": {
                    "resource_id": "inst-q00",
                    "port_id": "quel3-02-a01:tx_p04",
                    "role": "TRANSMITTER",
                }
            }
        },
        "BOX2": {
            "instruments": {
                "RQ00": {
                    "resource_id": "inst-rq00",
                    "port_id": "quel3-02-a02:trx_p00p04",
                    "role": "TRANSCEIVER",
                }
            }
        },
        "BOX3": {"instruments": {}},
    }


def test_sync_backend_settings_to_cache_restores_alias_mapping_from_snapshot() -> None:
    """Given hardware snapshot, cache sync should restore alias mappings."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    manager.sync_backend_settings_to_cache(
        backend_settings={
            "BOX1": {
                "instruments": {
                    "Q00": {
                        "resource_id": "inst-q00",
                        "port_id": "quel3-02-a01:tx_p04",
                        "role": "TRANSMITTER",
                    }
                }
            },
            "BOX2": {
                "instruments": {
                    "RQ00": {
                        "resource_id": "inst-rq00",
                        "port_id": "quel3-02-a02:trx_p00p04",
                        "role": "TRANSCEIVER",
                    }
                }
            },
        }
    )

    assert manager.target_alias_map == {"Q00": "Q00", "RQ00": "RQ00"}
    assert manager.last_deployed_instrument_infos["Q00"][0].id == "inst-q00"
    assert (
        manager.last_deployed_instrument_infos["Q00"][0].port_id
        == "quel3-02-a01:tx_p04"
    )
    assert manager.last_deployed_instrument_infos["Q00"][0].definition.alias == "Q00"
    assert (
        manager.last_deployed_instrument_infos["RQ00"][0].definition.role
        == "TRANSCEIVER"
    )


def test_deploy_instruments_replaces_cached_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given cached alias, deploy_instruments should replace it through quelware."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    cached_info = _CachedInstrumentInfo(
        id="inst-q00",
        port_id="quel3-02-a01:tx_p04",
        definition=_CachedDefinition(
            alias="Q00",
            role=_CachedRole(name="TRANSMITTER"),
            profile=_CachedProfile(
                frequency_range_min=4.0e9,
                frequency_range_max=4.5e9,
            ),
        ),
    )
    manager._last_deployed_instrument_infos = {"Q00": (cached_info,)}  # noqa: SLF001
    deploy_calls: list[tuple[str, list[object], bool]] = []
    returned_info = _CachedInstrumentInfo(
        id="inst-q00-new",
        port_id="quel3-02-a01:tx_p04",
        definition=_CachedDefinition(alias="Q00"),
    )

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[object],
            append: bool = False,
        ) -> list[object]:
            deploy_calls.append((port_id, definitions, append))
            return [returned_info]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            del resource_ids
            return _FakeSession()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    deployed = manager.deploy_instruments(
        requests=(
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
                target_labels=("Q00",),
            ),
        )
    )

    assert len(deploy_calls) == 1
    assert deploy_calls[0][2] is True
    assert deployed == {"Q00": (returned_info,)}
    assert manager.target_alias_map == {"Q00": "Q00"}


def test_deploy_instruments_replaces_cached_port_in_one_batched_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given cached port instruments, deploy should replace the port in one call."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    manager._last_deployed_instrument_infos = {  # noqa: SLF001
        "Q00": (
            _CachedInstrumentInfo(
                id="inst-q00",
                port_id="quel3-02-a01:tx_p04",
                definition=_CachedDefinition(
                    alias="Q00",
                    role=_CachedRole(name="TRANSMITTER"),
                    profile=_CachedProfile(
                        frequency_range_min=4.0e9,
                        frequency_range_max=4.5e9,
                    ),
                ),
            ),
        )
    }

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    deploy_calls: list[tuple[str, list[_Definition], bool]] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            deploy_calls.append((port_id, definitions, append))
            return [
                _InstrumentInfo(
                    id=f"inst:{definition.alias}",
                    port_id=port_id,
                    definition=definition,
                )
                for definition in definitions
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            del resource_ids
            return _FakeSession()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    deployed = manager.deploy_instruments(
        requests=(
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
                target_labels=("Q00",),
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q00-CR",
                target_labels=("Q00-CR",),
            ),
        )
    )

    assert len(deploy_calls) == 1
    assert deploy_calls[0][2] is True
    assert [definition.alias for definition in deploy_calls[0][1]] == ["Q00", "Q00-CR"]
    assert set(deployed) == {"Q00", "Q00-CR"}
