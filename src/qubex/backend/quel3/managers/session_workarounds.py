"""Workarounds for transient quelware-client session lifecycle failures."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Collection
from contextlib import AbstractAsyncContextManager, suppress
from typing import TYPE_CHECKING, TypeVar

from qubex.backend.quel3.interfaces.client import (
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)

if TYPE_CHECKING:
    from qubex.backend.quel3.managers.session_manager import Quel3SessionManager

T = TypeVar("T")

QUEL3_SESSION_REQUEST_MAX_ATTEMPTS = 4
QUELWARE_SESSION_CREATE_TTL_MS = 30_000
QUELWARE_SESSION_EXTEND_TTL_MS = 30_000
QUELWARE_SESSION_CREATE_TENTATIVE_TTL_MS = 5_000
QUELWARE_SESSION_CREATE_MAX_ATTEMPTS = 4
QUELWARE_SESSION_CREATE_INITIAL_RETRY_DELAY_SECONDS = 0.5
QUELWARE_SESSION_CREATE_MAX_RETRY_DELAY_SECONDS = 4.0
QUELWARE_SESSION_CREATE_RETRY_BACKOFF_FACTOR = 1.5
QUELWARE_SESSION_CREATE_RETRY_DELAY_SECONDS = (
    QUELWARE_SESSION_CREATE_INITIAL_RETRY_DELAY_SECONDS
)
QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS = 4

# Add diagnoses here using module-qualified exception class names.
# String keys keep quelware-client optional and avoid importing it for logging.
QUELWARE_EXCEPTION_HINTS: dict[str, str] = {
    "quelware_client.core.exceptions.LockConflictError": (
        "Another user may be using these resources, or one of your own sessions "
        "may still hold their locks. Check for sessions that have not been released."
    ),
}

# Exact HTTP status codes take precedence over status families such as "5xx".
QUELWARE_HTTP_STATUS_HINTS: dict[str, str] = {
    "413": (
        "The payload may be too large. Check whether an IQ array contains "
        "more than 65536 samples."
    ),
    "5xx": (
        "The QuEL server may have failed internally, possibly due to a server bug. "
        "Check server and proxy logs for this session."
    ),
}

logger = logging.getLogger(__name__)


class QuelwareSessionError(RuntimeError):
    """Runtime error annotated with the captured quelware session token."""

    def __init__(
        self,
        message: str,
        *,
        session_token: str,
        cause: BaseException,
    ) -> None:
        self.session_token = session_token
        super().__init__(
            f"{message}; session_token={session_token}; "
            f"cause={type(cause).__name__}: {cause}"
        )


async def run_with_session_request_retry(
    *,
    manager: Quel3SessionManager,
    client_factory: QuelwareClientFactory,
    resource_ids: tuple[ResourceIdProtocol, ...],
    operation: Callable[[SessionProtocol], Awaitable[T]],
) -> T:
    """
    Run one operation, recreating the client and session after request failures.

    Session creation retains its own retry budget. Successful sessions remain
    open until the next operation or the caller's batch cleanup.
    """
    max_attempts = max(1, int(QUEL3_SESSION_REQUEST_MAX_ATTEMPTS))
    for attempt in range(max_attempts):
        attempt_number = attempt + 1
        try:
            if not manager.is_open:
                await manager.open(client_factory=client_factory)
            session_token = manager.session_token or "<unavailable>"
            try:
                session = await manager.reopen_session(resource_ids)
            except QuelwareSessionError:
                raise
            except Exception as exc:
                raise QuelwareSessionError(
                    "QuEL-3 quelware session reopen failed",
                    session_token=session_token,
                    cause=exc,
                ) from exc
            if session is None:
                raise RuntimeError(  # noqa: TRY301
                    "QuEL-3 session reopen did not return an execution session."
                )
            logger.info(
                "QuEL-3 quelware session opened; session_token=%s; attempt=%d/%d",
                manager.session_token or "<unavailable>",
                attempt_number,
                max_attempts,
            )
            return await operation(session)
        except Exception as exc:
            session_token = (
                exc.session_token
                if isinstance(exc, QuelwareSessionError)
                else manager.session_token or "<unavailable>"
            )
            await manager.close_safely()
            if attempt_number >= max_attempts:
                hint = _exception_hint(exc)
                logger.exception(
                    "QuEL-3 quelware session request failed after retries; "
                    "session_token=%s; attempt=%d/%d%s",
                    session_token,
                    attempt_number,
                    max_attempts,
                    f"; possible cause: {hint}" if hint else "",
                )
                if isinstance(exc, QuelwareSessionError):
                    raise
                raise QuelwareSessionError(
                    "QuEL-3 quelware session request failed after retries",
                    session_token=session_token,
                    cause=exc,
                ) from exc
            logger.warning(
                "QuEL-3 quelware session request failed; session_token=%s; "
                "attempt=%d/%d; retrying with a fresh session; cause=%s",
                session_token,
                attempt_number,
                max_attempts,
                quelware_exception_summary(exc),
            )
    raise RuntimeError("unreachable QuEL-3 session request retry state")


def quelware_session_token(session: object | None) -> str:
    """Return a printable quelware session token for diagnostics."""
    if session is None:
        return "<unavailable>"
    try:
        token = getattr(session, "token", None)
    except Exception:
        return "<unavailable>"
    if token is None:
        return "<unavailable>"
    try:
        return str(token)
    except Exception:
        return "<unprintable>"


def quelware_exception_summary(exc: BaseException) -> str:
    """Return exception context with a user-defined possible cause when registered."""
    summary = f"{type(exc).__name__}: {exc}"
    hint = _exception_hint(exc)
    return f"{summary}; possible cause: {hint}" if hint else summary


def _exception_hint(exc: BaseException) -> str | None:
    """Find the first registered class hint along the explicit exception cause chain."""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        for cls in type(current).__mro__:
            class_name = f"{cls.__module__}.{cls.__qualname__}"
            hint = QUELWARE_EXCEPTION_HINTS.get(class_name)
            if hint:
                return hint
            if class_name in {"urllib.error.HTTPError", "grpclib.exceptions.GRPCError"}:
                hint = _http_status_hint(current)
                if hint:
                    return hint
        current = current.__cause__
    return None


def _http_status_hint(exc: BaseException) -> str | None:
    """Look up an HTTP status from an HTTPError or a gRPC transport error message."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        status = str(code)
    else:
        message = getattr(exc, "message", None)
        if not isinstance(message, str):
            return None
        match = re.search(r"(?:\bHTTP\s+|:status\s*=\s*['\"]?)([1-5]\d{2})\b", message)
        if match is None:
            return None
        status = match[1]
    return QUELWARE_HTTP_STATUS_HINTS.get(status) or QUELWARE_HTTP_STATUS_HINTS.get(
        f"{status[0]}xx"
    )


def is_resource_allocation_error(exc: BaseException) -> bool:
    """Return whether an exception looks like transient resource allocation failure."""
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    grpc_code = _grpc_code_name(exc)
    if grpc_code is not None:
        text = f"{text} {grpc_code.lower()}"
    if "unit" in text and ("unavailable" in text or "not ready" in text):
        return True
    if "resource" not in text:
        return False
    return any(
        keyword in text
        for keyword in (
            "acquire",
            "acquired",
            "allocate",
            "allocated",
            "allocation",
            "already",
            "available",
            "busy",
            "exhausted",
            "in use",
            "locked",
            "release",
            "released",
            "unavailable",
        )
    )


async def enter_quelware_session_with_resource_retry(
    *,
    client: QuelwareClientProtocol,
    resource_ids: Collection[ResourceIdProtocol],
) -> tuple[AbstractAsyncContextManager[SessionProtocol], SessionProtocol]:
    """
    Enter a quelware session while retrying transient resource allocation failures.

    Notes
    -----
    This is a QuEL-3-local workaround for observed `quelware-client` behavior
    where resources can briefly appear unavailable after the previous session
    released them.
    """
    normalized_resource_ids = tuple(resource_ids)
    max_attempts = max(1, int(QUELWARE_SESSION_CREATE_MAX_ATTEMPTS))
    for attempt in range(max_attempts):
        session_cm: AbstractAsyncContextManager[SessionProtocol] | None = None
        try:
            session_cm = client.create_session(
                normalized_resource_ids,
                ttl_ms=QUELWARE_SESSION_CREATE_TTL_MS,
                tentative_ttl_ms=QUELWARE_SESSION_CREATE_TENTATIVE_TTL_MS,
            )
            session = await session_cm.__aenter__()
        except Exception as exc:
            session_token = quelware_session_token(session_cm)
            if session_cm is not None:
                await _close_after_failed_enter(session_cm=session_cm, exc=exc)
            attempt_number = attempt + 1
            retryable = is_resource_allocation_error(exc)
            if attempt_number >= max_attempts or not retryable:
                raise QuelwareSessionError(
                    "QuEL-3 quelware session creation failed after retries",
                    session_token=session_token,
                    cause=exc,
                ) from exc
            retry_delay = _session_create_retry_delay(attempt)
            logger.warning(
                "QuEL-3 quelware session creation failed; session_token=%s; "
                "attempt=%d/%d; retrying in %.3g s; cause=%s",
                session_token,
                attempt_number,
                max_attempts,
                retry_delay,
                quelware_exception_summary(exc),
            )
            await asyncio.sleep(retry_delay)
        else:
            return session_cm, session
    raise RuntimeError("unreachable quelware session retry state")


async def _close_after_failed_enter(
    *,
    session_cm: AbstractAsyncContextManager[SessionProtocol],
    exc: Exception,
) -> None:
    """Close a context manager that failed partway through `__aenter__`."""
    with suppress(Exception):
        await session_cm.__aexit__(type(exc), exc, exc.__traceback__)


def _session_create_retry_delay(attempt: int) -> float:
    """Return retry delay for a failed session creation attempt."""
    delay = QUELWARE_SESSION_CREATE_INITIAL_RETRY_DELAY_SECONDS * (
        QUELWARE_SESSION_CREATE_RETRY_BACKOFF_FACTOR**attempt
    )
    return min(delay, QUELWARE_SESSION_CREATE_MAX_RETRY_DELAY_SECONDS)


def _grpc_code_name(exc: BaseException) -> str | None:
    """Return a gRPC status-code name when the exception exposes one."""
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        value = code()
    except Exception:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)
