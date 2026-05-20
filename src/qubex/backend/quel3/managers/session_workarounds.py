"""Workarounds for transient quelware-client session lifecycle failures."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection
from contextlib import AbstractAsyncContextManager, suppress

from qubex.backend.quel3.interfaces.client import (
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)

QUELWARE_SESSION_CREATE_MAX_ATTEMPTS = 4
QUELWARE_SESSION_CREATE_INITIAL_RETRY_DELAY_SECONDS = 0.5
QUELWARE_SESSION_CREATE_MAX_RETRY_DELAY_SECONDS = 4.0
QUELWARE_SESSION_CREATE_RETRY_BACKOFF_FACTOR = 1.5
QUELWARE_SESSION_CREATE_RETRY_DELAY_SECONDS = (
    QUELWARE_SESSION_CREATE_INITIAL_RETRY_DELAY_SECONDS
)
QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS = 4

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
    """Return one-line exception context for retry logs."""
    return f"{type(exc).__name__}: {exc}"


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
            session_cm = client.create_session(normalized_resource_ids)
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
