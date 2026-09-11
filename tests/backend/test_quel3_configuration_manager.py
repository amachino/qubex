"""Tests for QuEL-3 backend configuration manager behavior."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3HardwareStateReader,
    Quel3HttpTransportConfig,
    Quel3RuntimeConfig,
    configuration_manager as configuration_manager_module,
    runtime_config as runtime_config_module,
    session_workarounds as session_workarounds_module,
)
from qubex.backend.quel3.managers.session_workarounds import QuelwareSessionError
from qubex.backend.quel3.models import (
    InstrumentConfiguration,
    InstrumentRoleName,
    InstrumentSpec,
)


def _deploy_configuration(
    manager: Quel3ConfigurationManager,
    *,
    specifications: Sequence[InstrumentSpec],
    parallel: bool = True,
) -> dict[str, InstrumentInfoProtocol]:
    """Deploy with complete readback objects and a fresh real instrument cache."""
    configuration = InstrumentConfiguration(instruments=tuple(specifications))
    infos = tuple(
        cast(
            InstrumentInfoProtocol,
            SimpleNamespace(
                id=f"readback:{specification.alias}",
                port_id=specification.port_id,
                definition=SimpleNamespace(
                    alias=specification.alias,
                    role=specification.role,
                    mode="FIXED_TIMELINE",
                    profile=SimpleNamespace(
                        frequency_range_min=specification.frequency_range_min_hz,
                        frequency_range_max=specification.frequency_range_max_hz,
                    ),
                ),
                config=SimpleNamespace(sampling_period_fs=400_000),
            ),
        )
        for specification in specifications
    )
    cache = InstrumentCache()
    read_calls: list[tuple[str, ...]] = []

    def read_infos(
        *, port_ids: Sequence[str], parallel: bool
    ) -> tuple[InstrumentInfoProtocol, ...]:
        read_calls.append(tuple(port_ids))
        return infos

    reader = cast(
        Quel3HardwareStateReader,
        SimpleNamespace(read_instrument_infos=read_infos),
    )

    result = manager.deploy_instruments(
        configuration=configuration,
        instrument_cache=cache,
        hardware_state_reader=reader,
        parallel=parallel,
    )

    expected = {info.definition.alias: info for info in infos}
    assert result == expected
    assert cache.snapshot() == expected
    assert read_calls == (
        [tuple(dict.fromkeys(spec.port_id for spec in specifications))]
        if specifications
        else []
    )
    return result


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


@pytest.mark.parametrize(
    ("role", "expected_role"),
    [
        ("TRANSMITTER", "transmitter"),
        ("TRANSCEIVER", "transceiver"),
        ("TRANSCEIVER_LOOPBACK", "transceiver_loopback"),
        ("RECEIVER", "receiver"),
    ],
)
def test_deploy_instruments_calls_session_api(
    monkeypatch: pytest.MonkeyPatch,
    role: InstrumentRoleName,
    expected_role: str,
) -> None:
    """Given deploy specifications, backend configuration manager should call session deploy."""
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
        TRANSCEIVER_LOOPBACK = "transceiver_loopback"
        RECEIVER = "receiver"

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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role=role,
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    deployed = _deploy_configuration(manager, specifications=(specification,))

    assert create_session_calls == [("quel3-02-a01:tx_p02",)]
    assert len(deploy_calls) == 1
    port_id, definitions = deploy_calls[0]
    assert port_id == "quel3-02-a01:tx_p02"
    definition = definitions[0]
    assert definition.mode == "fixed_timeline"
    assert definition.role == expected_role
    assert definition.profile.frequency_range_min == pytest.approx(4.1e9)
    assert definition.profile.frequency_range_max == pytest.approx(4.3e9)
    assert definition.alias == "Q00"
    assert "Q00" in deployed


def test_deploy_instruments_recreates_session_after_transient_request_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient quelware specification failure, deploy should retry with a new session."""
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
                raise RuntimeError("quelware specification failed")
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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    deployed = _deploy_configuration(manager, specifications=(specification,))

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
    assert "Q00" in deployed


def test_deploy_instruments_finishes_parallel_ports_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one port fails, deploy should finish other ports before cleanup and retry."""
    manager = Quel3ConfigurationManager()
    closed_active_counts: list[int] = []
    deploy_calls: list[tuple[int, str]] = []

    class _FakeSession:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt
            self.active = 0

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            closed_active_counts.append(self.active)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[object],
            append: bool,
        ) -> list[object]:
            del definitions, append
            deploy_calls.append((self.attempt, port_id))
            if self.attempt == 0 and port_id == "unit:tx_p02":
                raise RuntimeError("First port failed")
            self.active += 1
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.active -= 1
            return []

    sessions: list[_FakeSession] = []

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
            assert tuple(resource_ids) == ("unit:tx_p02", "unit:tx_p04")
            session = _FakeSession(len(sessions))
            sessions.append(session)
            return session

    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: lambda endpoint, port: _FakeClient(),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: _make_instrument_entities(
            SimpleNamespace,
            SimpleNamespace,
            SimpleNamespace(FIXED_TIMELINE="fixed_timeline"),
            SimpleNamespace(TRANSMITTER="transmitter"),
        ),
    )

    _deploy_configuration(
        manager,
        specifications=tuple(
            InstrumentSpec(
                port_id=port_id,
                alias=alias,
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
            )
            for port_id, alias in (("unit:tx_p02", "Q00"), ("unit:tx_p04", "Q01"))
        ),
    )

    assert closed_active_counts == [0, 0]
    assert deploy_calls == [
        (0, "unit:tx_p02"),
        (0, "unit:tx_p04"),
        (1, "unit:tx_p02"),
        (1, "unit:tx_p04"),
    ]


def test_deploy_instruments_ignores_session_close_failure_after_success(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given deploy succeeds but close fails, deploy should complete without retrying."""
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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    deployed = _deploy_configuration(manager, specifications=(specification,))

    assert session.exit_calls == 1
    assert client.exit_calls == 1
    assert "QuEL-3 quelware deploy session cleanup failed" in caplog.text
    assert "cleanup-failed-deploy-session" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert "Q00" in deployed


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
            raise RuntimeError("quelware specification failed")

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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    with pytest.raises(
        QuelwareSessionError,
        match=f"session_token={failed_session_id}",
    ) as exc_info:
        _deploy_configuration(manager, specifications=(specification,))

    assert exc_info.value.session_token == failed_session_id
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "quelware specification failed"
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

        def create_session(self, resource_ids: list[str], **_: object) -> object:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    deployed = _deploy_configuration(manager, specifications=(specification,))

    assert client.create_session_calls == [
        ("quel3-02-a01:tx_p02",),
        ("quel3-02-a01:tx_p02",),
    ]
    assert client.failing_context.exit_calls == 1
    assert session.exit_calls == 1
    assert "Q00" in deployed


def test_deploy_instruments_ignores_returned_instrument_infos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no returned instrument infos, deploy should still apply the specification."""
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
            return []

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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specification = InstrumentSpec(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="Q00",
    )

    deployed = _deploy_configuration(manager, specifications=(specification,))

    assert [definition.alias for definition in deploy_definitions] == ["Q00"]
    assert "Q00" in deployed


def test_deploy_instruments_skips_client_for_empty_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given empty specifications, deploy should return without loading a client."""
    manager = Quel3ConfigurationManager()

    def fail_if_client_loaded() -> None:
        pytest.fail("Empty deployment should not load a quelware client.")

    monkeypatch.setattr(manager, "_load_quelware_client_factory", fail_if_client_loaded)

    assert _deploy_configuration(manager, specifications=()) == {}


def test_deploy_instruments_groups_requests_by_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given same-port specifications, backend configuration manager should batch one deploy call."""
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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specifications = (
        InstrumentSpec(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
        ),
        InstrumentSpec(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q00-CR",
        ),
    )

    deployed = _deploy_configuration(manager, specifications=specifications)

    assert create_session_calls == [("quel3-02-a01:tx_p04",)]
    assert len(deploy_calls) == 1
    assert deploy_calls[0][0] == "quel3-02-a01:tx_p04"
    assert deploy_calls[0][2] is False
    assert [definition.alias for definition in deploy_calls[0][1]] == [
        "Q00",
        "Q00-CR",
    ]
    assert "Q00" in deployed


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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    specifications = (
        InstrumentSpec(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
        ),
        InstrumentSpec(
            port_id="quel3-02-a01:tx_p06",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="Q01",
        ),
    )

    deployed = _deploy_configuration(manager, specifications=specifications)

    assert create_session_calls == [
        ("quel3-02-a01:tx_p04", "quel3-02-a01:tx_p06"),
    ]
    assert deploy_calls == ["quel3-02-a01:tx_p04", "quel3-02-a01:tx_p06"]
    assert "Q00" in deployed


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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    _deploy_configuration(
        manager,
        specifications=(
            InstrumentSpec(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
            ),
            InstrumentSpec(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
            ),
        ),
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

        def create_session(self, resource_ids: list[str], **_: object) -> _FakeSession:
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

    _deploy_configuration(
        manager,
        specifications=(
            InstrumentSpec(
                port_id="quel3-02-a01:tx_p04",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
                alias="Q00",
            ),
            InstrumentSpec(
                port_id="quel3-02-a01:tx_p06",
                role="TRANSMITTER",
                frequency_range_min_hz=4.2e9,
                frequency_range_max_hz=4.4e9,
                alias="Q01",
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
        lambda *, client_mode, pat_path=None, transport, http_transport: (
            captured.update(
                {
                    "client_mode": client_mode,
                    "pat_path": pat_path,
                    "transport": transport,
                    "http_transport": http_transport,
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
        "transport": "grpc",
        "http_transport": None,
    }


def test_load_client_factory_uses_configured_pat_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given PAT path runtime option, client factory loading should forward runtime settings."""
    captured: dict[str, object] = {}
    fake_client_factory = object()
    pat_path = "/run/secrets/quelware-pat"

    def _load_quelware_client_factory(
        *,
        client_mode: str,
        pat_path: str,
        transport: str,
        http_transport: Quel3HttpTransportConfig | None,
    ) -> object:
        captured["client_mode"] = client_mode
        captured["pat_path"] = pat_path
        captured["transport"] = transport
        captured["http_transport"] = http_transport
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
        "transport": "grpc",
        "http_transport": None,
    }
