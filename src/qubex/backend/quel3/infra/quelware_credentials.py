"""Credentials and lazy PAT resolution for quelware clients."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


class QuelwareCredentials:
    """
    Own PAT sources shared by quelware backend components.

    Providers take precedence over the PAT file. Neither source is evaluated
    during credentials construction. The upstream client evaluates the provider
    once at client creation and retains that PAT for its lifetime. Tokens and
    provider representations are not included in the credentials representation.
    """

    def __init__(
        self,
        *,
        pat_path: str | None = None,
        pat_provider: Callable[[], str] | None = None,
    ) -> None:
        """
        Configure optional PAT sources without accessing credentials.

        Parameters
        ----------
        pat_path : str | None, optional
            PAT file path. User-home shorthand is supported; trailing newline
            characters are removed each time the file is read.
        pat_provider : Callable[[], str] | None, optional
            Zero-argument callable returning the current PAT without processing.
            Takes precedence over `pat_path`.
        """
        self._pat_path = pat_path
        self._provider: Callable[[], str] | None = None
        if pat_provider is not None:
            self.set_pat_provider(pat_provider)

    @property
    def pat_path(self) -> str | None:
        """Return the configured PAT file path."""
        return self._pat_path

    def set_pat_provider(self, provider: Callable[[], str]) -> None:
        """
        Register a PAT provider without evaluating it.

        Recreate existing clients to apply the updated provider. Updating these
        shared credentials directly does not close or reauthenticate clients.

        Parameters
        ----------
        provider : Callable[[], str]
            Zero-argument callable returning the current PAT.

        Raises
        ------
        TypeError
            The provider is not callable.
        """
        if not callable(provider):
            raise TypeError("quelware PAT provider must be callable.")
        self._provider = provider

    def resolve_pat_provider(self) -> Callable[[], str] | None:
        """Return a lazy provider, or None to use upstream authentication defaults."""
        if self._provider is None and self._pat_path is None:
            return None
        return self._read_pat

    def _read_pat(self) -> str:
        if self._provider is not None:
            return self._provider()
        if self._pat_path is not None:
            return (
                Path(self._pat_path)
                .expanduser()
                .read_text(encoding="utf-8")
                .rstrip("\r\n")
            )
        raise RuntimeError("No quelware PAT source is configured.")
