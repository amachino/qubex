"""Execution-session manager for QuEL-3 quelware runtime."""

from __future__ import annotations

from collections.abc import Collection
from contextlib import AbstractAsyncContextManager
from types import TracebackType

from qubex.backend.quel3.infra.quelware_imports import (
    Quel3ClientMode,
    load_quelware_client_factory,
    validate_quelware_client_runtime,
)
from qubex.backend.quel3.interfaces import (
    QuelwareClientFactory,
    QuelwareClientProtocol,
    ResourceIdProtocol,
    SessionProtocol,
)
from qubex.backend.quel3.managers.session_workarounds import (
    QuelwareSessionError,
    enter_quelware_session_with_resource_retry,
    quelware_session_token,
)


class Quel3SessionManager:
    """Manage one open quelware client/session pair for execution reuse."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
        client_mode: str = "server",
        quelware_pat_path: str | None = None,
    ) -> None:
        normalized_client_mode = validate_quelware_client_runtime(
            client_mode=client_mode,
        )
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._client_mode: Quel3ClientMode = normalized_client_mode
        self._quelware_pat_path = quelware_pat_path
        self._client_cm: AbstractAsyncContextManager[QuelwareClientProtocol] | None = (
            None
        )
        self._client: QuelwareClientProtocol | None = None
        self._session_cm: AbstractAsyncContextManager[SessionProtocol] | None = None
        self._session: SessionProtocol | None = None
        self._session_token: str | None = None
        self._resource_ids: tuple[ResourceIdProtocol, ...] | None = None

    @property
    def quelware_endpoint(self) -> str:
        """Return quelware endpoint."""
        return self._quelware_endpoint

    @property
    def quelware_port(self) -> int:
        """Return quelware port."""
        return self._quelware_port

    @property
    def client_mode(self) -> Quel3ClientMode:
        """Return configured quelware client mode."""
        return self._client_mode

    @property
    def quelware_pat_path(self) -> str | None:
        """Return configured quelware personal access token path."""
        return self._quelware_pat_path

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

    async def open(
        self,
        resource_ids: Collection[ResourceIdProtocol] | None = None,
        *,
        client_factory: QuelwareClientFactory | None = None,
    ) -> SessionProtocol | None:
        """
        Open quelware client/session resources.

        If `resource_ids` is omitted, only the client context is opened.
        If the client is already open and the session matches the requested
        resources, the existing session is reused.
        """
        if self._client is None:
            runtime_client_factory = (
                self.load_quelware_client_factory()
                if client_factory is None
                else client_factory
            )
            self._client_cm = runtime_client_factory(
                self._quelware_endpoint,
                self._quelware_port,
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
            return self._session

        try:
            (
                self._session_cm,
                self._session,
            ) = await enter_quelware_session_with_resource_retry(
                client=self._client,
                resource_ids=normalized_resource_ids,
            )
        except Exception:
            await self.close()
            raise
        self._session_token = quelware_session_token(self._session)
        self._resource_ids = normalized_resource_ids
        return self._session

    async def reopen_session(
        self,
        resource_ids: Collection[ResourceIdProtocol] | None = None,
    ) -> SessionProtocol | None:
        """Close the current session and open a replacement on the same client."""
        session_cm, session_token = self._detach_session_context()
        if session_cm is not None:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception as exc:
                raise QuelwareSessionError(
                    "QuEL-3 quelware session close failed before reopening",
                    session_token=session_token,
                    cause=exc,
                ) from exc
        return await self.open(resource_ids)

    async def close(self) -> None:
        """Close any open quelware session and client contexts."""
        session_cm, _ = self._detach_session_context()
        try:
            if session_cm is not None:
                await session_cm.__aexit__(None, None, None)
        finally:
            client_cm = self._client_cm
            self._client_cm = None
            self._client = None

            if client_cm is not None:
                await client_cm.__aexit__(None, None, None)

    def _detach_session_context(
        self,
    ) -> tuple[AbstractAsyncContextManager[SessionProtocol] | None, str]:
        """Clear stored session state and return the previous context."""
        session_cm = self._session_cm
        session_token = self._session_token or quelware_session_token(self._session)
        self._session_cm = None
        self._session = None
        self._session_token = None
        self._resource_ids = None
        return session_cm, session_token

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
        return load_quelware_client_factory(
            client_mode=self._client_mode,
            pat_path=self._quelware_pat_path,
        )
