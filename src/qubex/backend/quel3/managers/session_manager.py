"""Execution-session manager for QuEL-3 quelware runtime."""

from __future__ import annotations

import asyncio
import logging
from asyncio import sleep
from collections.abc import Collection
from contextlib import AbstractAsyncContextManager
from types import TracebackType

from qubex.backend.quel3.infra.quelware_imports import Quel3ClientMode
from qubex.backend.quel3.infra.runtime_config import Quel3RuntimeConfig
from qubex.backend.quel3.interfaces import (
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.managers.session_workarounds import (
    QUELWARE_SESSION_EXTEND_TTL_MS,
    QuelwareSessionAcquisitionError,
    QuelwareSessionError,
    enter_quelware_session,
    quelware_exception_summary,
    quelware_session_token,
)

logger = logging.getLogger(__name__)

QUEL3_LEASE_RENEW_INTERVAL_SECONDS = 5.0


class Quel3SessionManager:
    """Manage one open quelware client/session pair for execution reuse."""

    def __init__(
        self,
        *,
        runtime_config: Quel3RuntimeConfig | None = None,
    ) -> None:
        self._runtime_config = runtime_config or Quel3RuntimeConfig()
        self._client_cm: AbstractAsyncContextManager[QuelwareClientProtocol] | None = (
            None
        )
        self._client: QuelwareClientProtocol | None = None
        self._session_cm: AbstractAsyncContextManager[SessionProtocol] | None = None
        self._session: SessionProtocol | None = None
        self._session_token: str | None = None
        self._resource_ids: tuple[ResourceIdProtocol, ...] | None = None
        self._renewal_task: asyncio.Task[None] | None = None
        self._lease_expires_at = 0.0

    @property
    def runtime_config(self) -> Quel3RuntimeConfig:
        """Return the shared quelware runtime config."""
        return self._runtime_config

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint."""
        return self._runtime_config.endpoint

    @property
    def quelware_port(self) -> int | None:
        """Return quelware port."""
        return self._runtime_config.port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._runtime_config.client_mode_value

    @property
    def is_open(self) -> bool:
        """Return whether a client context is currently open."""
        return self._client is not None

    @property
    def client(self) -> QuelwareClientProtocol:
        """Return the currently open quelware client."""
        if self._client is None:
            raise RuntimeError("QuEL-3 session manager is not open.")
        return self._client

    @property
    def session(self) -> SessionProtocol:
        """Return the currently open quelware session."""
        if self._session is None:
            raise RuntimeError("QuEL-3 execution session is not open.")
        return self._session

    @property
    def session_token(self) -> str | None:
        """Return the token captured when the current session opened."""
        return self._session_token

    @property
    def resource_ids(self) -> tuple[ResourceIdProtocol, ...] | None:
        """Return resource IDs bound to the current open session."""
        return self._resource_ids

    def has_valid_lease(self, session: SessionProtocol) -> bool:
        """Return whether this session still has a confirmed execution lease."""
        return (
            self._session is session
            and asyncio.get_running_loop().time() < self._lease_expires_at
        )

    async def open(
        self,
        resource_ids: Collection[ResourceIdProtocol] | None = None,
        *,
        client_factory: QuelwareClientFactory | None = None,
    ) -> SessionProtocol | None:
        """
        Open quelware client/session resources.

        If `resource_ids` is omitted, only the client context is opened.
        An existing session with the same resources and a confirmed lease is
        reused. Each acquired session owns one periodic lease renewal task.
        """
        if self._client is None:
            runtime_client_factory = (
                self.load_quelware_client_factory()
                if client_factory is None
                else client_factory
            )
            self._client_cm = runtime_client_factory(
                self._runtime_config.endpoint,
                self._runtime_config.port,
            )
            try:
                self._client = await self._client_cm.__aenter__()
            except Exception:
                self._client_cm = None
                raise

        if resource_ids is None:
            return None

        normalized_resource_ids = tuple(resource_ids)
        if self._session is not None:
            if normalized_resource_ids != self._resource_ids:
                raise RuntimeError(
                    "QuEL-3 session manager is already bound to different resources."
                )
            if self.has_valid_lease(self._session):
                return self._session
            return await self.reopen_session(normalized_resource_ids)

        try:
            (
                self._session_cm,
                self._session,
            ) = await enter_quelware_session(
                client=self._client,
                resource_ids=normalized_resource_ids,
            )
            self._session_token = quelware_session_token(self._session)
            started = asyncio.get_running_loop().time()
            if not await self._session.extend(QUELWARE_SESSION_EXTEND_TTL_MS):
                raise RuntimeError("Initial session lease extension was rejected.")  # noqa: TRY301 - Handle rejection and RPC failure together.
        except (QuelwareSessionAcquisitionError, asyncio.CancelledError):
            await self.close_safely()
            raise
        except Exception as exc:
            token = self._session_token or "<unavailable>"
            await self.close_safely()
            raise QuelwareSessionAcquisitionError(
                "QuEL-3 lease extension failed after session acquisition",
                session_token=token,
                cause=exc,
            ) from exc
        self._resource_ids = normalized_resource_ids
        self._lease_expires_at = started + QUELWARE_SESSION_EXTEND_TTL_MS / 1000
        self._renewal_task = asyncio.create_task(
            self._renew_lease(self._session, started)
        )
        return self._session

    async def _renew_lease(self, session: SessionProtocol, opened_at: float) -> None:
        """Renew the acquired session until it closes, expires, or rejects renewal."""
        loop = asyncio.get_running_loop()
        sent = opened_at
        while self.has_valid_lease(session):
            await sleep(max(0, sent + QUEL3_LEASE_RENEW_INTERVAL_SECONDS - loop.time()))
            sent = loop.time()
            if sent >= self._lease_expires_at:
                break
            try:
                accepted = await session.extend(QUELWARE_SESSION_EXTEND_TTL_MS)
            except Exception as exc:
                logger.warning(
                    "QuEL-3 lease renewal failed; session_token=%s; cause=%s",
                    self._session_token,
                    quelware_exception_summary(exc),
                )
                continue
            if not accepted:
                self._lease_expires_at = 0.0
                logger.warning(
                    "QuEL-3 lease renewal rejected; session_token=%s",
                    self._session_token,
                )
                return
            self._lease_expires_at = sent + QUELWARE_SESSION_EXTEND_TTL_MS / 1000
        logger.warning(
            "QuEL-3 session lease expired; session_token=%s", self._session_token
        )

    async def reopen_session(
        self,
        resource_ids: Collection[ResourceIdProtocol] | None = None,
    ) -> SessionProtocol | None:
        """Close the current session and open a replacement on the same client."""
        session_token = self._session_token or "<unavailable>"
        try:
            await self._close_session()
        except Exception as exc:
            raise QuelwareSessionError(
                "QuEL-3 quelware session close failed before reopening",
                session_token=session_token,
                cause=exc,
            ) from exc
        return await self.open(resource_ids)

    async def close_safely(self) -> None:
        """Close session state without masking request success or failure."""
        session_token = self.session_token or "<unavailable>"
        try:
            await self.close()
        except Exception as exc:
            logger.warning(
                "QuEL-3 quelware session cleanup failed; session_token=%s; cause=%s",
                session_token,
                quelware_exception_summary(exc),
            )

    async def close(self) -> None:
        """Close any open quelware session and client contexts."""
        try:
            await self._close_session()
        finally:
            client_cm = self._client_cm
            self._client_cm = None
            self._client = None

            if client_cm is not None:
                await client_cm.__aexit__(None, None, None)

    async def _close_session(self) -> None:
        """Stop renewal before releasing the associated session context."""
        session_cm = self._session_cm
        renewal = self._renewal_task
        self._renewal_task = None
        self._session_cm = None
        self._session = None
        self._session_token = None
        self._resource_ids = None
        try:
            if renewal is not None:
                renewal.cancel()
                await asyncio.gather(renewal, return_exceptions=True)
        finally:
            if session_cm is not None:
                await session_cm.__aexit__(None, None, None)

    async def __aenter__(self) -> Quel3SessionManager:
        """Open the underlying quelware client context and return self."""
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close any open session/client context on async context exit."""
        del exc_type, exc, tb
        await self.close()

    def load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Import quelware client factory lazily."""
        return self._runtime_config.load_client_factory()
