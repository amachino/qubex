"""Tests for QuEL-3 session manager behavior."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from qubex.backend.quel3.managers import (
    Quel3SessionManager,
    session_workarounds as session_workarounds_module,
)
from qubex.backend.quel3.managers.session_workarounds import (
    is_resource_allocation_error,
)


class _SuccessfulSession:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self) -> _SuccessfulSession:
        self.enter_calls += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1


class _FailingSessionContext:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.exit_calls = 0

    async def __aenter__(self) -> object:
        raise self._exc

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1


class _FakeClient:
    def __init__(
        self,
        successful_session: _SuccessfulSession,
        *,
        failure: Exception | None = None,
        failures_before_success: int = 2,
    ) -> None:
        self._successful_session = successful_session
        self._failure = failure
        self._failures_before_success = failures_before_success
        self.failing_contexts: list[_FailingSessionContext] = []
        self.create_session_calls: list[tuple[str, ...]] = []

    def create_session(self, resource_ids: tuple[str, ...]) -> object:
        self.create_session_calls.append(tuple(resource_ids))
        if len(self.create_session_calls) <= self._failures_before_success:
            context = _FailingSessionContext(
                self._failure or RuntimeError("resource is not available yet")
            )
            self.failing_contexts.append(context)
            return context
        return self._successful_session


class _InvalidUnitStatusError(Exception):
    pass


class _FakeClientContext:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.exit_calls = 0

    async def __aenter__(self) -> _FakeClient:
        return self._client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1


def _patch_session_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch session retry sleep and return recorded delays."""
    delays: list[float] = []

    async def _record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(session_workarounds_module.asyncio, "sleep", _record_sleep)
    return delays


def test_open_retries_transient_resource_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient resource allocation failure, open should retry session creation."""
    session = _SuccessfulSession()
    client = _FakeClient(successful_session=session)
    client_context = _FakeClientContext(client)
    sleep_delays = _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    async def _run() -> object:
        opened_session = await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close()
        return opened_session

    opened_session = asyncio.run(_run())

    assert opened_session is session
    assert client.create_session_calls == [("inst-a",), ("inst-a",), ("inst-a",)]
    assert [context.exit_calls for context in client.failing_contexts] == [1, 1]
    assert sleep_delays == pytest.approx([0.5, 0.75])
    assert session.enter_calls == 1
    assert session.exit_calls == 1
    assert client_context.exit_calls == 1


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("resource is not available yet"),
        _InvalidUnitStatusError(
            "Some units are not ready to open new session. "
            "status: ({'quel3-02-a01': 'UnitStatus.UNAVAILABLE'})"
        ),
    ],
)
def test_session_error_classifier_accepts_transient_quelware_open_failures(
    exc: Exception,
) -> None:
    """Given known transient quelware open failures, classifier should accept them."""
    assert is_resource_allocation_error(exc) is True


def test_open_retries_transient_unit_unavailable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient unit unavailable failure, open should retry session creation."""
    session = _SuccessfulSession()
    client = _FakeClient(
        successful_session=session,
        failure=_InvalidUnitStatusError(
            "Some units are not ready to open new session. "
            "status: ({'quel3-02-a01': 'UnitStatus.UNAVAILABLE'})"
        ),
    )
    client_context = _FakeClientContext(client)
    sleep_delays = _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    async def _run() -> object:
        opened_session = await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close()
        return opened_session

    opened_session = asyncio.run(_run())

    assert opened_session is session
    assert client.create_session_calls == [("inst-a",), ("inst-a",), ("inst-a",)]
    assert [context.exit_calls for context in client.failing_contexts] == [1, 1]
    assert sleep_delays == pytest.approx([0.5, 0.75])
    assert session.exit_calls == 1


def test_open_stops_after_session_create_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given sustained unit unavailable failure, open should stop at retry limit."""
    session = _SuccessfulSession()
    client = _FakeClient(
        successful_session=session,
        failure=_InvalidUnitStatusError(
            "Some units are not ready to open new session. "
            "status: ({'quel3-02-a01': 'UnitStatus.UNAVAILABLE'})"
        ),
        failures_before_success=5,
    )
    client_context = _FakeClientContext(client)
    sleep_delays = _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    async def _run() -> None:
        with pytest.raises(_InvalidUnitStatusError):
            await manager.open(
                ("inst-a",),
                client_factory=cast(Any, lambda endpoint, port: client_context),
            )
        await manager.close()

    asyncio.run(_run())

    assert client.create_session_calls == [("inst-a",)] * 4
    assert [context.exit_calls for context in client.failing_contexts] == [
        1,
        1,
        1,
        1,
    ]
    assert sleep_delays == pytest.approx([0.5, 0.75, 1.125])
    assert session.exit_calls == 0
    assert client_context.exit_calls == 1


def test_open_succeeds_on_final_session_create_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given final retry succeeds, open should return the session."""
    session = _SuccessfulSession()
    client = _FakeClient(
        successful_session=session,
        failure=_InvalidUnitStatusError(
            "Some units are not ready to open new session. "
            "status: ({'quel3-02-a01': 'UnitStatus.UNAVAILABLE'})"
        ),
        failures_before_success=3,
    )
    client_context = _FakeClientContext(client)
    sleep_delays = _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    async def _run() -> object:
        opened_session = await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close()
        return opened_session

    opened_session = asyncio.run(_run())

    assert opened_session is session
    assert client.create_session_calls == [("inst-a",)] * 4
    assert [context.exit_calls for context in client.failing_contexts] == [
        1,
        1,
        1,
    ]
    assert sleep_delays == pytest.approx([0.5, 0.75, 1.125])
    assert session.exit_calls == 1
