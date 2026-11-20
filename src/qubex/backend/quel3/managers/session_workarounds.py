"""Workarounds for transient quelware-client session lifecycle failures."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Collection
from contextlib import AbstractAsyncContextManager, suppress
from typing import TYPE_CHECKING, TypeVar

from qubex.backend.quel3.instrument_cache import INSTRUMENT_RECOVERY_HINT
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
QUEL3_SESSION_REQUEST_RETRY_DELAY_SECONDS = 0.5
QUELWARE_SESSION_CREATE_TTL_MS = 30_000
QUELWARE_SESSION_EXTEND_TTL_MS = 30_000
QUELWARE_SESSION_REQUEST_MAX_ATTEMPTS = 4

# Add diagnoses here using module-qualified exception class names.
# String keys keep quelware-client optional and avoid importing it for logging.
QUELWARE_EXCEPTION_HINTS: dict[str, str] = {
    "quelware_client.core.exceptions.ResourceNotFoundError": INSTRUMENT_RECOVERY_HINT,
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
        hint = (
            f" {INSTRUMENT_RECOVERY_HINT}"
            if is_resource_not_found_error(cause)
            and INSTRUMENT_RECOVERY_HINT not in f"{message} {cause}"
            else ""
        )
        super().__init__(
            f"{message}; session_token={session_token}; "
            f"cause={type(cause).__name__}: {cause}{hint}"
        )


class QuelwareSessionAcquisitionError(QuelwareSessionError):
    """The server could not grant a usable session."""


class Quel3ExecutionOutcomeUnknown(QuelwareSessionError):
    """A dispatched trigger whose outcome could not be confirmed; never replay it."""


async def run_with_session_request_retry(
    *,
    manager: Quel3SessionManager,
    client_factory: QuelwareClientFactory,
    resource_ids: tuple[ResourceIdProtocol, ...],
    operation: Callable[[SessionProtocol], Awaitable[T]],
    reuse_session: bool = False,
) -> T:
    """
    Run an operation, retrying request failures with a fresh client and session.

    `reuse_session` keeps an open session when its resource IDs match.
    Otherwise, open a new session. The server handles acquisition retries.
    Missing resources propagate to the caller for cache refresh.
    """
    max_attempts = max(1, int(QUEL3_SESSION_REQUEST_MAX_ATTEMPTS))
    for attempt in range(max_attempts):
        attempt_number = attempt + 1
        try:
            started = time.monotonic()
            if not manager.is_open:
                await manager.open(client_factory=client_factory)
            session_token = manager.session_token or "<unavailable>"
            try:
                session = (
                    await manager.open(resource_ids)
                    if reuse_session and manager.resource_ids == resource_ids
                    else await manager.reopen_session(resource_ids)
                )
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
            logger.debug(
                "QuEL-3 phase=session opened; session_token=%s; attempt=%d/%d; elapsed_seconds=%.6f",
                manager.session_token or "<unavailable>",
                attempt_number,
                max_attempts,
                time.monotonic() - started,
            )
            return await operation(session)
        except (Quel3ExecutionOutcomeUnknown, QuelwareSessionAcquisitionError):
            raise
        except Exception as exc:
            session_token = (
                exc.session_token
                if isinstance(exc, QuelwareSessionError)
                else manager.session_token or "<unavailable>"
            )
            await manager.close_safely()
            if is_resource_not_found_error(exc):
                if isinstance(exc, QuelwareSessionError):
                    raise
                raise QuelwareSessionError(
                    "QuEL-3 quelware resource not found",
                    session_token=session_token,
                    cause=exc,
                ) from exc
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
                "attempt=%d/%d; retrying with a fresh session in %.3g s; cause=%s",
                session_token,
                attempt_number,
                max_attempts,
                QUEL3_SESSION_REQUEST_RETRY_DELAY_SECONDS,
                quelware_exception_summary(exc),
            )
            await asyncio.sleep(QUEL3_SESSION_REQUEST_RETRY_DELAY_SECONDS)
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
    return (
        f"{summary}; possible cause: {hint}"
        if hint and hint not in summary
        else summary
    )


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


def is_resource_not_found_error(exc: BaseException) -> bool:
    """Identify typed missing resources along the explicit, cycle-safe cause chain."""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if any(
            f"{cls.__module__}.{cls.__qualname__}"
            == "quelware_client.core.exceptions.ResourceNotFoundError"
            for cls in type(current).__mro__
        ):
            return True
        current = current.__cause__
    return False


async def enter_quelware_session(
    *,
    client: QuelwareClientProtocol,
    resource_ids: Collection[ResourceIdProtocol],
) -> tuple[AbstractAsyncContextManager[SessionProtocol], SessionProtocol]:
    """Acquire a session using the server's retry window and tentative lock TTL."""
    session_cm = None
    try:
        session_cm = client.create_session(
            tuple(resource_ids), ttl_ms=QUELWARE_SESSION_CREATE_TTL_MS
        )
        session = await session_cm.__aenter__()
    except (Exception, asyncio.CancelledError) as exc:
        token = quelware_session_token(session_cm)
        if session_cm is not None:
            with suppress(Exception):
                await session_cm.__aexit__(type(exc), exc, exc.__traceback__)
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.error(
            "QuEL-3 session acquisition failed; session_token=%s; cause=%s",
            token,
            quelware_exception_summary(exc),
            exc_info=True,
        )
        raise QuelwareSessionAcquisitionError(
            "QuEL-3 quelware session acquisition failed", session_token=token, cause=exc
        ) from exc
    return session_cm, session
