"""Tests for QuEL-3 session manager behavior."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest

from qubex.backend.quel3.managers import (
    Quel3SessionManager,
    session_workarounds as session_workarounds_module,
)
from qubex.backend.quel3.managers.session_workarounds import (
    QuelwareSessionError,
    is_resource_allocation_error,
    quelware_exception_summary,
    quelware_session_token,
)


class _SuccessfulSession:
    def __init__(self, *, session_id: str = "successful-session") -> None:
        self.token = session_id
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
    def __init__(self, exc: Exception, *, session_id: str) -> None:
        self.token = session_id
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

    def create_session(self, resource_ids: tuple[str, ...], **_: object) -> object:
        self.create_session_calls.append(tuple(resource_ids))
        if len(self.create_session_calls) <= self._failures_before_success:
            context = _FailingSessionContext(
                self._failure or RuntimeError("resource is not available yet"),
                session_id=f"failed-create-session-{len(self.create_session_calls)}",
            )
            self.failing_contexts.append(context)
            return context
        return self._successful_session


class _InvalidUnitStatusError(Exception):
    pass


class _UnopenedSession:
    @property
    def token(self) -> str:
        raise ValueError("Token not found. Session may not opened.")


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


def test_session_token_helper_handles_unopened_session_property() -> None:
    """Given unopened session token raises, token helper should return unavailable."""
    assert quelware_session_token(_UnopenedSession()) == "<unavailable>"


def test_lock_conflict_hint_mentions_other_users_and_unreleased_sessions():
    """A lock conflict should suggest both other users and the caller's unreleased sessions."""
    error_type = type(
        "LockConflictError",
        (Exception,),
        {"__module__": "quelware_client.core.exceptions"},
    )
    summary = quelware_exception_summary(error_type("locked"))
    assert "Another user" in summary
    assert "your own sessions" in summary
    assert "not been released" in summary


@pytest.mark.parametrize("kind", ["direct", "subclass", "wrapped"])
def test_exception_summary_uses_user_defined_hints(monkeypatch, kind):
    """Registered hints should apply to exception classes, subclasses, and explicit causes."""
    error_type = type(
        "LockConflictError",
        (Exception,),
        {"__module__": "quelware_client.core.exceptions"},
    )
    monkeypatch.setitem(
        session_workarounds_module.QUELWARE_EXCEPTION_HINTS,
        "quelware_client.core.exceptions.LockConflictError",
        "user-defined cause",
    )
    if kind == "subclass":
        error_type = type("SpecializedError", (error_type,), {})
    cause = error_type("original message")
    error = cause
    if kind == "wrapped":
        error = RuntimeError("request failed")
        error.__cause__ = cause

    summary = quelware_exception_summary(error)

    assert (
        summary
        == f"{type(error).__name__}: {error}; possible cause: user-defined cause"
    )
    assert str(cause) == "original message"
    if kind == "wrapped":
        assert error.__cause__ is cause


def test_exception_summary_leaves_unregistered_errors_unchanged(monkeypatch):
    """Unregistered classes and cyclic causes should retain the original exception summary."""
    monkeypatch.setitem(
        session_workarounds_module.QUELWARE_EXCEPTION_HINTS,
        "quelware_client.core.exceptions.LockConflictError",
        "user-defined cause",
    )
    unknown = RuntimeError("unknown")
    unknown.__cause__ = unknown
    assert quelware_exception_summary(unknown) == "RuntimeError: unknown"
    lookalike = type("LockConflictError", (Exception,), {})("locked")
    assert quelware_exception_summary(lookalike) == "LockConflictError: locked"


def test_exception_hint_does_not_change_retry_classification(monkeypatch):
    """Hint text should not affect resource-allocation retry classification."""
    monkeypatch.setitem(
        session_workarounds_module.QUELWARE_EXCEPTION_HINTS,
        "builtins.RuntimeError",
        "resource unavailable",
    )
    error = RuntimeError("unknown failure")
    assert "resource unavailable" in quelware_exception_summary(error)
    assert not is_resource_allocation_error(error)


@pytest.mark.parametrize("fail_close", [False, True])
def test_close_safely_releases_resources_and_logs_cleanup_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, fail_close: bool
) -> None:
    """Safe close should release the client and report cleanup failure with the saved token."""
    changed_session_id = "changed-during-close"
    monkeypatch.setitem(
        session_workarounds_module.QUELWARE_EXCEPTION_HINTS,
        "builtins.RuntimeError",
        "user-defined cleanup cause",
    )

    class ClosingSession(_SuccessfulSession):
        async def __aexit__(self, exc_type, exc, tb):
            await super().__aexit__(exc_type, exc, tb)
            self.token = changed_session_id
            if fail_close:
                raise RuntimeError("session close failed")

    session = ClosingSession(session_id="saved-session")
    client = _FakeClient(successful_session=session, failures_before_success=0)
    client_context = _FakeClientContext(client)
    manager = Quel3SessionManager()

    async def run() -> None:
        await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close_safely()

    asyncio.run(run())

    assert session.exit_calls == client_context.exit_calls == 1
    assert not manager.is_open
    assert manager.session_token is None
    if fail_close:
        assert "session_token=saved-session" in caplog.text
        assert "session close failed" in caplog.text
        assert "changed-during-close" not in caplog.text
        assert "possible cause: user-defined cleanup cause" in caplog.text
    else:
        assert not caplog.records


def test_open_retries_transient_resource_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient resource allocation failure, open should retry session creation."""
    session = _SuccessfulSession()
    client = _FakeClient(successful_session=session)
    client_context = _FakeClientContext(client)
    sleep_delays = _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager()

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


@pytest.mark.parametrize("creation_failures", [3, 4])
def test_request_retry_preserves_separate_session_creation_budget(
    monkeypatch: pytest.MonkeyPatch, creation_failures: int
) -> None:
    """Each request attempt should retain its own session creation retry budget."""
    delays = _patch_session_retry_sleep(monkeypatch)
    contexts: list[_FakeClientContext] = []
    clients: list[_FakeClient] = []
    manager = Quel3SessionManager()
    result = object()
    calls = 0

    def factory(endpoint, port):
        client = _FakeClient(
            successful_session=_SuccessfulSession(),
            failures_before_success=creation_failures,
        )
        clients.append(client)
        context = _FakeClientContext(client)
        contexts.append(context)
        return context

    async def operation(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("request failed")
        return result

    async def run():
        try:
            return await session_workarounds_module.run_with_session_request_retry(
                manager=manager,
                client_factory=cast(Any, factory),
                resource_ids=("inst-a",),
                operation=operation,
            )
        finally:
            await manager.close_safely()

    if creation_failures == 3:
        assert asyncio.run(run()) is result
        assert calls == len(clients) == 2
    else:
        with pytest.raises(QuelwareSessionError, match="session creation failed"):
            asyncio.run(run())
        assert calls == 0
        assert len(clients) == 4
    assert all(len(client.create_session_calls) == 4 for client in clients)
    assert all(context.exit_calls == 1 for context in contexts)
    assert delays == pytest.approx([0.5, 0.75, 1.125] * len(clients))


def test_request_retry_preserves_previous_session_token_on_reopen_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed close before reopening should retain the previous session token and cause."""
    error_type = type(
        "LockNotFoundError",
        (Exception,),
        {"__module__": "quelware_client.core.exceptions"},
    )
    cause = error_type("close before reopening failed")
    monkeypatch.setitem(
        session_workarounds_module.QUELWARE_EXCEPTION_HINTS,
        "quelware_client.core.exceptions.LockNotFoundError",
        "user-defined reopening cause",
    )
    expected_session_id = "previous-session"

    class ClosingSession(_SuccessfulSession):
        async def __aexit__(self, exc_type, exc, tb):
            raise cause

    client = _FakeClient(
        successful_session=ClosingSession(session_id=expected_session_id),
        failures_before_success=0,
    )
    context = _FakeClientContext(client)
    factory = cast(Any, lambda endpoint, port: context)
    manager = Quel3SessionManager()
    monkeypatch.setattr(
        session_workarounds_module, "QUEL3_SESSION_REQUEST_MAX_ATTEMPTS", 1
    )

    async def operation(session):
        pytest.fail("Execution must not start when session reopening fails.")

    async def run():
        await manager.open(("inst-a",), client_factory=factory)
        await session_workarounds_module.run_with_session_request_retry(
            manager=manager,
            client_factory=factory,
            resource_ids=("inst-a",),
            operation=operation,
        )

    with pytest.raises(QuelwareSessionError) as error:
        asyncio.run(run())
    assert error.value.session_token == expected_session_id
    assert error.value.__cause__ is cause
    assert context.exit_calls == 1
    assert "possible cause: user-defined reopening cause" in caplog.text


def test_open_retries_when_failed_session_token_is_unavailable(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given failed session has no token yet, open should preserve the original failure."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.session_workarounds",
    )
    session = _SuccessfulSession()
    failure = RuntimeError("resource is not available yet")

    class _FailingUnopenedSession(_UnopenedSession):
        def __init__(self) -> None:
            self.exit_calls = 0

        async def __aenter__(self) -> object:
            raise failure

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            del exc_type, exc, tb
            self.exit_calls += 1

    class _FakeClientWithUnopenedFailure:
        def __init__(self) -> None:
            self.failing_context = _FailingUnopenedSession()
            self.create_session_calls: list[tuple[str, ...]] = []

        def create_session(self, resource_ids: tuple[str, ...], **_: object) -> object:
            self.create_session_calls.append(tuple(resource_ids))
            if len(self.create_session_calls) == 1:
                return self.failing_context
            return session

    client = _FakeClientWithUnopenedFailure()
    client_context = _FakeClientContext(cast(Any, client))
    _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager()

    async def _run() -> object:
        opened_session = await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close()
        return opened_session

    opened_session = asyncio.run(_run())

    assert opened_session is session
    assert client.create_session_calls == [("inst-a",), ("inst-a",)]
    assert client.failing_context.exit_calls == 1
    assert "session_token=<unavailable>" in caplog.text
    assert "resource is not available yet" in caplog.text
    assert "Token not found" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_open_logs_session_token_on_session_create_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given session creation fails, open should log the failed session token."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.session_workarounds",
    )
    session = _SuccessfulSession()
    client = _FakeClient(successful_session=session, failures_before_success=1)
    client_context = _FakeClientContext(client)
    _patch_session_retry_sleep(monkeypatch)
    manager = Quel3SessionManager()

    async def _run() -> None:
        await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        await manager.close()

    asyncio.run(_run())

    assert "QuEL-3 quelware session creation failed" in caplog.text
    assert "failed-create-session-1" in caplog.text
    assert "attempt=1/4" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_open_captures_session_token_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given session opens, manager should store its token immediately."""
    session = _SuccessfulSession(session_id="opened-session")
    client = _FakeClient(successful_session=session, failures_before_success=0)
    client_context = _FakeClientContext(client)
    manager = Quel3SessionManager()

    async def _run() -> tuple[str | None, str | None]:
        await manager.open(
            ("inst-a",),
            client_factory=cast(Any, lambda endpoint, port: client_context),
        )
        captured_session_id = manager.session_token
        changed_session_id = "mutated-session"
        session.token = changed_session_id
        await manager.close()
        return captured_session_id, manager.session_token

    captured_session_id, closed_session_id = asyncio.run(_run())

    assert captured_session_id == "opened-session"
    assert closed_session_id is None


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
    manager = Quel3SessionManager()

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
    expected_session_id = "failed-create-session-4"
    manager = Quel3SessionManager()

    async def _run() -> QuelwareSessionError:
        with pytest.raises(
            QuelwareSessionError,
            match=f"session_token={expected_session_id}",
        ) as exc_info:
            await manager.open(
                ("inst-a",),
                client_factory=cast(Any, lambda endpoint, port: client_context),
            )
        await manager.close()
        return exc_info.value

    error = asyncio.run(_run())

    assert error.session_token == expected_session_id
    assert isinstance(error.__cause__, _InvalidUnitStatusError)
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
    manager = Quel3SessionManager()

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
