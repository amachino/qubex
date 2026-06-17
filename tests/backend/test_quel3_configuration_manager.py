"""Tests for QuEL-3 backend configuration manager behavior."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3RuntimeConfig,
    configuration_manager as configuration_manager_module,
    runtime_config as runtime_config_module,
    session_workarounds as session_workarounds_module,
)
from qubex.backend.quel3.managers.session_workarounds import QuelwareSessionError
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
    mode: object | None = None
    role: object | None = None
    profile: _CachedProfile | None = None


@dataclass(frozen=True)
class _CachedInstrumentInfo:
    id: str
    port_id: str
    definition: _CachedDefinition


def _make_instrument_entities(
    profile_factory: Any,
    definition_factory: Any,
    mode_namespace: Any,
    role_namespace: Any,
) -> Any:
    """Create one fake instrument-entity boundary for configuration tests."""
    return configuration_manager_module._QuelwareInstrumentEntities(  # noqa: SLF001
        fixed_timeline_profile_factory=profile_factory,
        instrument_definition_factory=definition_factory,
        instrument_mode_namespace=mode_namespace,
        instrument_role_namespace=role_namespace,
    )


def test_deploy_instruments_calls_session_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given deploy requests, backend configuration manager should call session deploy."""
    manager = Quel3ConfigurationManager()

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
            assert append is False
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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
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
    assert manager.target_alias_map == {("BOX1", "Q00"): "quel3-02-a01:Q00"}
    assert definition.alias in deployed


def test_deploy_instruments_recreates_session_after_transient_request_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient quelware request failure, deploy should retry with a new session."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.configuration_manager",
    )
    manager = Quel3ConfigurationManager()

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

    class _FakeSession:
        def __init__(
            self,
            *,
            fail_once: bool,
            session_id: str,
            failed_session_id: str | None = None,
        ) -> None:
            self.token = session_id
            self._fail_once = fail_once
            self._failed_session_id = failed_session_id
            self.deploy_calls: list[str] = []
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            del append
            self.deploy_calls.append(port_id)
            if self._fail_once:
                self._fail_once = False
                if self._failed_session_id is not None:
                    self.token = self._failed_session_id
                raise RuntimeError("quelware request failed")
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}",
                    port_id=port_id,
                    definition=definitions[0],
                )
            ]

    class _FakeClient:
        def __init__(self, session: _FakeSession) -> None:
            self._session = session
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            assert tuple(resource_ids) == ("quel3-02-a01:tx_p02",)
            return self._session

    sessions = [
        _FakeSession(
            fail_once=True,
            session_id="failed-deploy-session",
            failed_session_id="mutated-deploy-session",
        ),
        _FakeSession(fail_once=False, session_id="retry-deploy-session"),
    ]
    clients: list[_FakeClient] = []

    def _create_client(endpoint: str, port: int) -> _FakeClient:
        del endpoint, port
        client = _FakeClient(sessions[len(clients)])
        clients.append(client)
        return client

    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: _create_client
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert len(clients) == 2
    assert [client.exit_calls for client in clients] == [1, 1]
    assert [session.exit_calls for session in sessions] == [1, 1]
    assert [session.deploy_calls for session in sessions] == [
        ["quel3-02-a01:tx_p02"],
        ["quel3-02-a01:tx_p02"],
    ]
    assert "QuEL-3 quelware deploy request failed" in caplog.text
    assert "failed-deploy-session" in caplog.text
    assert "mutated-deploy-session" not in caplog.text
    assert "retry-deploy-session" not in caplog.text
    assert "attempt=1/4" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert set(deployed) == {"Q00"}


def test_deploy_instruments_ignores_session_close_failure_after_success(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given deploy succeeds but close fails, deploy should preserve the result."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.configuration_manager",
    )
    manager = Quel3ConfigurationManager()

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

    class _FakeSession:
        def __init__(self, *, session_id: str) -> None:
            self.token = session_id
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1
            raise RuntimeError("quelware close failed")

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            del append
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}",
                    port_id=port_id,
                    definition=definitions[0],
                )
            ]

    class _FakeClient:
        def __init__(self, session: _FakeSession) -> None:
            self._session = session
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            assert tuple(resource_ids) == ("quel3-02-a01:tx_p02",)
            return self._session

    session = _FakeSession(session_id="cleanup-failed-deploy-session")
    client = _FakeClient(session)
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: client,
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert session.exit_calls == 1
    assert client.exit_calls == 1
    assert "QuEL-3 quelware deploy session cleanup failed" in caplog.text
    assert "cleanup-failed-deploy-session" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert set(deployed) == {"Q00"}


def test_deploy_instruments_wraps_final_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given retry limit is reached, deploy should raise a token-annotated error."""
    failed_session_id = "failed-deploy-session"
    manager = Quel3ConfigurationManager()

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

    class _FakeSession:
        def __init__(self) -> None:
            self.token = failed_session_id
            self.deploy_calls: list[str] = []
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[object]:
            _ = (definitions, append)
            self.deploy_calls.append(port_id)
            raise RuntimeError("quelware request failed")

    class _FakeClient:
        def __init__(self, session: _FakeSession) -> None:
            self._session = session
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            assert tuple(resource_ids) == ("quel3-02-a01:tx_p02",)
            return self._session

    session = _FakeSession()
    client = _FakeClient(session)
    monkeypatch.setattr(
        configuration_manager_module,
        "QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS",
        1,
    )
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: client,
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
    )

    with pytest.raises(
        QuelwareSessionError,
        match=f"session_token={failed_session_id}",
    ) as exc_info:
        manager.deploy_instruments(requests=(request,))

    assert exc_info.value.session_token == failed_session_id
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "quelware request failed"
    assert session.deploy_calls == ["quel3-02-a01:tx_p02"]
    assert session.exit_calls == 1
    assert client.exit_calls == 1


def test_deploy_instruments_retries_resource_allocation_on_session_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient resource allocation failure, deploy should retry session creation."""
    manager = Quel3ConfigurationManager()

    async def _skip_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(session_workarounds_module.asyncio, "sleep", _skip_sleep)

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

    class _FailingSessionContext:
        def __init__(self) -> None:
            self.exit_calls = 0

        async def __aenter__(self) -> object:
            raise RuntimeError("resource is not available yet")

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

    class _FakeSession:
        def __init__(self) -> None:
            self.exit_calls = 0

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)
            self.exit_calls += 1

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[_Definition],
            append: bool = False,
        ) -> list[_InstrumentInfo]:
            del append
            return [
                _InstrumentInfo(
                    id=f"id:{port_id}",
                    port_id=port_id,
                    definition=definitions[0],
                )
            ]

    class _FakeClient:
        def __init__(self, session: _FakeSession) -> None:
            self._session = session
            self.failing_context = _FailingSessionContext()
            self.create_session_calls: list[tuple[str, ...]] = []

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> object:
            self.create_session_calls.append(tuple(resource_ids))
            if len(self.create_session_calls) == 1:
                return self.failing_context
            return self._session

    session = _FakeSession()
    client = _FakeClient(session)
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: client,
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert client.create_session_calls == [
        ("quel3-02-a01:tx_p02",),
        ("quel3-02-a01:tx_p02",),
    ]
    assert client.failing_context.exit_calls == 1
    assert session.exit_calls == 1
    assert set(deployed) == {"Q00"}


def test_deploy_instruments_accepts_unit_prefixed_returned_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware prefixes aliases by unit, deploy should keep target bindings usable."""
    manager = Quel3ConfigurationManager()

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

    deploy_definitions: list[_Definition] = []

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
            del append
            deploy_definitions.extend(definitions)
            returned_definition = _Definition(
                alias="quel3-02-a01:Q00",
                mode=definitions[0].mode,
                role=definitions[0].role,
                profile=definitions[0].profile,
            )
            return [
                _InstrumentInfo(
                    id="inst-q00",
                    port_id=port_id,
                    definition=returned_definition,
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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert [definition.alias for definition in deploy_definitions] == ["Q00"]
    assert set(deployed) == {"Q00"}
    assert deployed["Q00"][0].definition.alias == "quel3-02-a01:Q00"
    assert manager.target_alias_map == {("BOX1", "Q00"): "quel3-02-a01:Q00"}


def test_deploy_instruments_clears_cache_for_empty_requests() -> None:
    """Given empty requests, backend configuration manager should clear deployment cache."""
    manager = Quel3ConfigurationManager()
    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
        target_labels=("Q00",),
        box_id="BOX1",
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
    manager._target_alias_map = {("BOX1", "Q00"): request.alias}  # noqa: SLF001

    deployed = manager.deploy_instruments(requests=())

    assert deployed == {}
    assert manager.last_deployed_instrument_infos == {}
    assert manager.target_alias_map == {}


def test_deploy_instruments_groups_requests_by_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given same-port requests, backend configuration manager should batch one deploy call."""
    manager = Quel3ConfigurationManager()

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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
            box_id="BOX1",
        ),
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q00-CR",
            target_labels=("Q00-CR",),
            box_id="BOX1",
        ),
    )

    deployed = manager.deploy_instruments(requests=requests)

    assert create_session_calls == [("quel3-02-a01:tx_p04",)]
    assert len(deploy_calls) == 1
    assert deploy_calls[0][0] == "quel3-02-a01:tx_p04"
    assert deploy_calls[0][2] is False
    assert [definition.alias for definition in deploy_calls[0][1]] == [
        "Q00",
        "Q00-CR",
    ]
    assert manager.target_alias_map == {
        ("BOX1", "Q00"): "quel3-02-a01:Q00",
        ("BOX1", "Q00-CR"): "quel3-02-a01:Q00-CR",
    }
    assert set(deployed) == {"Q00", "Q00-CR"}
    assert deployed["Q00"][0].id == "id:quel3-02-a01:tx_p04:0"
    assert deployed["Q00-CR"][0].id == "id:quel3-02-a01:tx_p04:1"


def test_deploy_instruments_uses_one_session_for_all_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given multiple ports, backend configuration manager should reuse one session."""
    manager = Quel3ConfigurationManager()

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
            assert append is False
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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
    )

    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
            box_id="BOX1",
        ),
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p06",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q01",
            target_labels=("Q01",),
            box_id="BOX1",
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
    manager = Quel3ConfigurationManager()

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
            assert append is False
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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
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
                box_id="BOX1",
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
                target_labels=("Q01",),
                box_id="BOX1",
            ),
        )
    )

    assert probe.max_active == 2


def test_deploy_instruments_parallel_false_serializes_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel false, deploy_instruments should deploy port batches serially."""
    manager = Quel3ConfigurationManager()

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
            assert append is False
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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
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
                box_id="BOX1",
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
                target_labels=("Q01",),
                box_id="BOX1",
            ),
        ),
        parallel=False,
    )

    assert probe.max_active == 1


def test_load_client_factory_uses_configured_client_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given server runtime options, client factory loading should use that runtime."""
    captured: dict[str, object] = {}
    fake_client_factory = object()
    monkeypatch.setattr(
        runtime_config_module,
        "load_quelware_client_factory",
        lambda *, client_mode, pat_path=None: (
            captured.update(
                {
                    "client_mode": client_mode,
                    "pat_path": pat_path,
                }
            )
            or fake_client_factory
        ),
    )
    manager = Quel3ConfigurationManager(
        runtime_config=Quel3RuntimeConfig(endpoint="worker-host", port=61000),
    )

    client_factory = manager._load_quelware_client_factory()  # noqa: SLF001

    assert client_factory is fake_client_factory
    assert captured == {
        "client_mode": "server",
        "pat_path": None,
    }


def test_load_client_factory_uses_configured_pat_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given PAT path runtime option, client factory loading should forward only the path."""
    captured: dict[str, object] = {}
    fake_client_factory = object()
    pat_path = "/run/secrets/quelware-pat"

    def _load_quelware_client_factory(
        *,
        client_mode: str,
        pat_path: str,
    ) -> object:
        captured["client_mode"] = client_mode
        captured["pat_path"] = pat_path
        return fake_client_factory

    monkeypatch.setattr(
        runtime_config_module,
        "load_quelware_client_factory",
        _load_quelware_client_factory,
    )
    manager = Quel3ConfigurationManager(
        runtime_config=Quel3RuntimeConfig(
            endpoint="worker-host",
            port=61000,
            pat_path=pat_path,
        ),
    )

    client_factory = manager._load_quelware_client_factory()  # noqa: SLF001

    assert client_factory is fake_client_factory
    assert captured == {
        "client_mode": "server",
        "pat_path": pat_path,
    }


def test_refresh_instrument_cache_loads_existing_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given existing quelware instruments, refreshing cache should expose alias mappings."""
    manager = Quel3ConfigurationManager()

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
    assert manager.target_alias_map == {}
    assert set(manager.last_deployed_instrument_infos.keys()) == {"Q00", "RQ00"}


def test_refresh_instrument_cache_maps_unit_prefixed_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given cached quelware aliases with unit prefixes, refresh should map local targets to runtime aliases."""
    manager = Quel3ConfigurationManager()

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
            return [_ResourceInfo("inst-q00")]

        async def get_instrument_info(self, resource_id: str) -> object:
            assert resource_id == "inst-q00"
            return _InstrumentInfo("quel3-02-a01:Q00", "quel3-02-a01:tx_p04")

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )

    cached = manager.refresh_instrument_cache()

    assert set(cached) == {"Q00"}
    assert manager.target_alias_map == {}
    assert manager.last_deployed_instrument_infos["Q00"][0].definition.alias == (
        "quel3-02-a01:Q00"
    )


def test_fetch_backend_settings_from_hardware_groups_instruments_by_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware instruments, hardware fetch should normalize them per box."""
    manager = Quel3ConfigurationManager()

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

    class _Mode:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Profile:
        def __init__(
            self,
            *,
            frequency_range_min: float,
            frequency_range_max: float,
        ) -> None:
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(
            self,
            alias: str,
            role: object,
            *,
            mode: object,
            profile: object,
        ) -> None:
            self.alias = alias
            self.role = role
            self.mode = mode
            self.profile = profile

    class _InstrumentInfo:
        def __init__(
            self,
            resource_id: str,
            alias: str,
            port_id: str,
            role: str,
            *,
            frequency_range_min: float,
            frequency_range_max: float,
        ) -> None:
            self.id = resource_id
            self.port_id = port_id
            self.definition = _Definition(
                alias,
                _Role(role),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=frequency_range_min,
                    frequency_range_max=frequency_range_max,
                ),
            )

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
                    frequency_range_min=4.1e9,
                    frequency_range_max=4.3e9,
                ),
                "inst-rq00": _InstrumentInfo(
                    "inst-rq00",
                    "RQ00",
                    "quel3-02-a02:trx_p00p04",
                    "TRANSCEIVER",
                    frequency_range_min=5.9e9,
                    frequency_range_max=6.1e9,
                ),
                "inst-other": _InstrumentInfo(
                    "inst-other",
                    "Q99",
                    "quel3-02-a99:tx_p01",
                    "TRANSMITTER",
                    frequency_range_min=4.0e9,
                    frequency_range_max=4.5e9,
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
                    "definition": {
                        "alias": "Q00",
                        "role": "TRANSMITTER",
                        "mode": "FIXED_TIMELINE",
                        "profile": {
                            "frequency_range_min": 4.1e9,
                            "frequency_range_max": 4.3e9,
                        },
                    },
                }
            }
        },
        "BOX2": {
            "instruments": {
                "RQ00": {
                    "resource_id": "inst-rq00",
                    "port_id": "quel3-02-a02:trx_p00p04",
                    "role": "TRANSCEIVER",
                    "definition": {
                        "alias": "RQ00",
                        "role": "TRANSCEIVER",
                        "mode": "FIXED_TIMELINE",
                        "profile": {
                            "frequency_range_min": 5.9e9,
                            "frequency_range_max": 6.1e9,
                        },
                    },
                }
            }
        },
        "BOX3": {"instruments": {}},
    }


def test_fetch_backend_settings_skips_unselected_unit_instrument_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given selected unit labels, hardware fetch should not inspect instruments on other units."""
    manager = Quel3ConfigurationManager()

    class _InstrumentCategory:
        name = "INSTRUMENT"

    class _ResourceInfo:
        def __init__(self, resource_id: str) -> None:
            self.id = resource_id
            self.category = _InstrumentCategory()

    class _Role:
        name = "TRANSMITTER"

    class _Definition:
        alias = "Q00"
        role = _Role()
        mode = None
        profile = None

    class _InstrumentInfo:
        id = "quel3-02-a01:inst-q00"
        port_id = "quel3-02-a01:tx_p04"
        definition = _Definition()

    get_calls: list[str] = []

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
                _ResourceInfo("quel3-02-a01:inst-q00"),
                _ResourceInfo("quel3-02-a22:inst-q22"),
            ]

        async def get_instrument_info(self, resource_id: str) -> object:
            get_calls.append(resource_id)
            if resource_id != "quel3-02-a01:inst-q00":
                raise AssertionError(f"Unexpected instrument fetch: {resource_id}")
            return _InstrumentInfo()

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )

    fetched = manager.fetch_backend_settings_from_hardware(
        unit_labels_by_box_id={"BOX1": "quel3-02-a01"},
    )

    assert get_calls == ["quel3-02-a01:inst-q00"]
    assert fetched["BOX1"]["instruments"]["Q00"]["resource_id"] == (
        "quel3-02-a01:inst-q00"
    )


def test_sync_backend_settings_to_cache_restores_alias_mapping_from_snapshot() -> None:
    """Given hardware snapshot, cache sync should restore alias mappings."""
    manager = Quel3ConfigurationManager()

    manager.sync_backend_settings_to_cache(
        backend_settings={
            "BOX1": {
                "instruments": {
                    "Q00": {
                        "resource_id": "inst-q00",
                        "port_id": "quel3-02-a01:tx_p04",
                        "role": "TRANSMITTER",
                        "definition": {
                            "alias": "Q00",
                            "role": "TRANSMITTER",
                            "mode": "FIXED_TIMELINE",
                            "profile": {
                                "frequency_range_min": 4.1e9,
                                "frequency_range_max": 4.3e9,
                            },
                        },
                    }
                }
            },
            "BOX2": {
                "instruments": {
                    "RQ00": {
                        "resource_id": "inst-rq00",
                        "port_id": "quel3-02-a02:trx_p00p04",
                        "role": "TRANSCEIVER",
                        "definition": {
                            "alias": "RQ00",
                            "role": "TRANSCEIVER",
                            "mode": "FIXED_TIMELINE",
                            "profile": {
                                "frequency_range_min": 5.9e9,
                                "frequency_range_max": 6.1e9,
                            },
                        },
                    }
                }
            },
        }
    )

    assert manager.target_alias_map == {
        ("BOX1", "Q00"): "quel3-02-a01:Q00",
        ("BOX2", "RQ00"): "quel3-02-a02:RQ00",
    }
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
    assert (
        manager.last_deployed_instrument_infos["Q00"][0].definition.mode
        == "FIXED_TIMELINE"
    )
    profile = manager.last_deployed_instrument_infos["Q00"][0].definition.profile
    assert profile is not None
    assert profile.frequency_range_min == pytest.approx(4.1e9)
    assert profile.frequency_range_max == pytest.approx(4.3e9)


def test_deploy_instruments_replaces_cached_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given cached alias, deploy_instruments should replace it through quelware."""
    manager = Quel3ConfigurationManager()

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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
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
                box_id="BOX1",
            ),
        )
    )

    assert len(deploy_calls) == 1
    assert deploy_calls[0][2] is False
    assert deployed == {"Q00": (returned_info,)}
    assert manager.target_alias_map == {("BOX1", "Q00"): "quel3-02-a01:Q00"}


def test_deploy_instruments_replaces_cached_port_in_one_batched_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given cached port instruments, deploy should replace the port in one call."""
    manager = Quel3ConfigurationManager()

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
        lambda: _make_instrument_entities(_Profile, _Definition, _Mode, _Role),
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
                box_id="BOX1",
            ),
            InstrumentDeployRequest(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q00-CR",
                target_labels=("Q00-CR",),
                box_id="BOX1",
            ),
        )
    )

    assert len(deploy_calls) == 1
    assert deploy_calls[0][2] is False
    assert [definition.alias for definition in deploy_calls[0][1]] == ["Q00", "Q00-CR"]
    assert set(deployed) == {"Q00", "Q00-CR"}
