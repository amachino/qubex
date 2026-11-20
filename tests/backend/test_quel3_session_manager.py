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
    quelware_exception_summary,
    quelware_session_token,
)


class _SuccessfulSession:
    def __init__(self, *, session_id: str = "successful-session") -> None:
        self.token = session_id
        self.enter_calls = 0
        self.exit_calls = 0

    async def extend(self, new_ttl_ms: int) -> bool:
        return True

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
    ) -> None:
        self._successful_session = successful_session
        self._failure = failure
        self.failing_contexts: list[_FailingSessionContext] = []
        self.create_session_calls: list[tuple[str, ...]] = []

    def create_session(self, resource_ids: tuple[str, ...], **_: object) -> object:
        self.create_session_calls.append(tuple(resource_ids))
        if self._failure is not None:
            context = _FailingSessionContext(
                self._failure,
                session_id=f"failed-create-session-{len(self.create_session_calls)}",
            )
            self.failing_contexts.append(context)
            return context
        return self._successful_session


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
def test_resource_not_found_preserves_cause_and_includes_recovery_advice(kind):
    """Missing resources should expose recovery advice in the exception itself."""
    error_type = type(
        "ResourceNotFoundError",
        (Exception,),
        {"__module__": "quelware_client.core.exceptions"},
    )
    if kind == "subclass":
        error_type = type("SpecializedMissingResource", (error_type,), {})
    cause = error_type("missing instrument unit-a:old")
    if kind == "wrapped":
        original = cause
        cause = RuntimeError("wrapped resource failure")
        cause.__cause__ = original
    session_id = "failed-session"
    error = QuelwareSessionError(
        "session failed", session_token=session_id, cause=cause
    )
    assert "exp.configure()" in str(error)
    assert "exp.backend_controller.get_resource_snapshot()" in str(error)
    assert "another user" in str(error)
    assert "session_token=failed-session" in str(error)
    assert str(cause) in str(error)


@pytest.mark.parametrize(
    "name", ["LockNotFoundError", "UnitNotFoundError", "RuntimeError"]
)
def test_other_missing_resources_do_not_suggest_instrument_reconfiguration(name):
    """Lock and unit failures should retain their own diagnostics."""
    error_type = type(
        name, (Exception,), {"__module__": "quelware_client.core.exceptions"}
    )
    cause = error_type("resource not found")
    cause.__cause__ = cause
    session_id = "failed-session"
    error = QuelwareSessionError(
        "session failed", session_token=session_id, cause=cause
    )
    assert "exp.configure()" not in str(error)


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
    client = _FakeClient(successful_session=session)
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


@pytest.mark.parametrize("error_name", ["LockConflictError", "UnitUnreachableError"])
def test_acquisition_failure_ends_request_after_one_open(monkeypatch, error_name):
    """A server-side acquisition failure should close both contexts and end the request."""
    errors = pytest.importorskip("quelware_client.core.exceptions")
    cause = getattr(errors, error_name)("resource unavailable")
    client = _FakeClient(_SuccessfulSession(), failure=cause)
    context = _FakeClientContext(client)
    delays = _patch_session_retry_sleep(monkeypatch)

    async def operation(session):
        pytest.fail("An unacquired session cannot execute a request.")

    with pytest.raises(QuelwareSessionError) as error:
        asyncio.run(
            session_workarounds_module.run_with_session_request_retry(
                manager=Quel3SessionManager(),
                client_factory=cast(Any, lambda *_: context),
                resource_ids=("inst-a",),
                operation=operation,
            )
        )
    assert error.value.__cause__ is cause
    assert error.value.session_token == client.failing_contexts[0].token
    assert client.create_session_calls == [("inst-a",)]
    assert client.failing_contexts[0].exit_calls == context.exit_calls == 1
    assert delays == []


@pytest.mark.parametrize("fail_first_request", [False, True])
def test_request_logs_only_retry_warnings_at_info_level(caplog, fail_first_request):
    """Successful requests should be quiet at INFO while retries warn before execution."""
    caplog.set_level(logging.INFO, logger=session_workarounds_module.__name__)
    session = _SuccessfulSession()
    client = _FakeClient(successful_session=session)
    context = _FakeClientContext(client)
    manager = Quel3SessionManager()
    result = object()
    calls = 0

    async def operation(opened_session):
        nonlocal calls
        calls += 1
        assert opened_session is session
        if fail_first_request and calls == 1:
            raise RuntimeError("temporary request failure")
        if fail_first_request:
            assert len(caplog.records) == 1
            assert caplog.records[0].levelno == logging.WARNING
            assert "temporary request failure" in caplog.text
            assert "retrying with a fresh session" in caplog.text
            assert "session_token=successful-session" in caplog.text
        else:
            assert not caplog.records
        return result

    async def run():
        try:
            return await session_workarounds_module.run_with_session_request_retry(
                manager=manager,
                client_factory=cast(Any, lambda endpoint, port: context),
                resource_ids=("inst-a",),
                operation=operation,
            )
        finally:
            await manager.close_safely()

    assert asyncio.run(run()) is result
    assert calls == (2 if fail_first_request else 1)
    assert len(caplog.records) == int(fail_first_request)


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


def test_open_captures_session_token_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given session opens, manager should store its token immediately."""
    session = _SuccessfulSession(session_id="opened-session")
    client = _FakeClient(successful_session=session)
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


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("acquisition_seconds", [0, 31])
def test_open_extends_lease_before_return(monkeypatch, accepted, acquisition_seconds):
    """Every newly acquired session should be extended before execution."""
    now = 100.0
    extensions = []

    class SlowSession(_SuccessfulSession):
        async def __aenter__(self):
            nonlocal now
            now += acquisition_seconds
            return await super().__aenter__()

        async def extend(self, new_ttl_ms):
            extensions.append(new_ttl_ms)
            return accepted

    async def run():
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now)
        session = SlowSession()
        context = _FakeClientContext(_FakeClient(session))
        manager = Quel3SessionManager()
        acquisition = manager.open(
            ("inst",), client_factory=cast(Any, lambda *_: context)
        )
        if accepted:
            opened = await acquisition
            assert opened is not None
            assert manager.has_valid_lease(opened)
            await manager.close()
        else:
            with pytest.raises(QuelwareSessionError):
                await acquisition
        assert extensions == [30_000]
        assert session.exit_calls == context.exit_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["close", "reopen", "close_error"])
def test_session_renews_until_stopped_before_exit(monkeypatch, ending):
    """One renewal task should survive reuse and stop before its session exits."""
    offset = 0.0
    attempts = asyncio.Queue()
    stopped = asyncio.Event()

    class Session(_SuccessfulSession):
        calls = 0

        async def extend(self, new_ttl_ms):
            self.calls += 1
            if self.calls == 1:
                return True
            attempts.put_nowait(new_ttl_ms)
            if self.calls == 2:
                raise ConnectionError("temporary renewal failure")
            if self.calls == 3:
                return True
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def __aexit__(self, *args):
            assert stopped.is_set()
            await super().__aexit__(*args)
            if ending == "close_error":
                raise RuntimeError("close failed")

    async def run():
        nonlocal offset
        loop = asyncio.get_running_loop()
        original_time = loop.time
        monkeypatch.setattr(loop, "time", lambda: original_time() + offset)
        manager = Quel3SessionManager()
        session = Session()
        client = _FakeClient(successful_session=session)
        ctx = _FakeClientContext(client)
        opened = await manager.open(("inst",), client_factory=cast(Any, lambda *_: ctx))
        assert await manager.open(("inst",)) is opened
        # Synchronous setup may occupy the loop before the renewal task first runs.
        offset += 4
        await asyncio.sleep(0)
        assert attempts.empty()
        for increment in (1.01, 5.01, 5.01):
            offset += increment
            assert await asyncio.wait_for(attempts.get(), timeout=0.5) == 30_000
        if ending == "reopen":
            await manager.reopen_session()
        await manager.close_safely()
        assert session.exit_calls == 1
        assert ctx.exit_calls == 1
        assert stopped.is_set()

    asyncio.run(run())
