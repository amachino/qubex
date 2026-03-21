"""Execution-session manager for QuEL-3 quelware runtime."""

from __future__ import annotations

from collections.abc import Collection
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


class Quel3SessionManager:
    """Manage one open quelware client/session pair for execution reuse."""

    def __init__(
        self,
        *,
        quelware_endpoint: str,
        quelware_port: int,
        client_mode: Quel3ClientMode = "server",
        standalone_unit_label: str | None = None,
    ) -> None:
        normalized_client_mode = validate_quelware_client_runtime(
            client_mode=client_mode,
            standalone_unit_label=standalone_unit_label,
        )
        self._quelware_endpoint = quelware_endpoint
        self._quelware_port = quelware_port
        self._client_mode: Quel3ClientMode = normalized_client_mode
        self._standalone_unit_label = standalone_unit_label
        self._client_cm = None
        self._client: QuelwareClientProtocol | None = None
        self._session_cm = None
        self._session: SessionProtocol | None = None
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
    def standalone_unit_label(self) -> str | None:
        """Return configured standalone unit label."""
        return self._standalone_unit_label

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
            self._client = await self._client_cm.__aenter__()

        if resource_ids is None:
            return None

        normalized_resource_ids = tuple(resource_ids)
        if self._session is not None:
            if normalized_resource_ids != self._resource_ids:
                raise RuntimeError(
                    "QuEL-3 session manager is already bound to different resources."
                )
            return self._session

        self._session_cm = self._client.create_session(normalized_resource_ids)
        self._session = await self._session_cm.__aenter__()
        self._resource_ids = normalized_resource_ids
        return self._session

    async def close(self) -> None:
        """Close any open quelware session and client contexts."""
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        self._session_cm = None
        self._session = None
        self._resource_ids = None

        if self._client_cm is not None:
            await self._client_cm.__aexit__(None, None, None)
        self._client_cm = None
        self._client = None

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
            standalone_unit_label=self._standalone_unit_label,
        )
