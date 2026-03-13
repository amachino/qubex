"""Fit result models and compatibility helpers."""

from __future__ import annotations

import warnings
from collections import UserDict
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import plotly.graph_objects as go

_LEGACY_FIT_FIGURE_KEYS = {
    "fig": "the `figure` attribute",
    "figure": "the `figure` attribute",
    "figs": "the `figures` attribute",
    "figures": "the `figures` attribute",
    "fig3d": "the `figures` attribute",
}


class FitStatus(Enum):
    """Status values for fit operations."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class _FitResultData(dict[str, object]):
    """
    Payload mapping for `FitResult`.

    This wrapper preserves mapping-style access while warning when callers read
    legacy figure keys from the payload.
    """

    def __getitem__(self, key: str) -> object:
        """Return one payload value and warn for legacy figure keys."""
        self._warn_if_legacy(key)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:
        """Return one payload value with the default when missing."""
        self._warn_if_legacy(key)
        return super().get(key, default)

    def _warn_if_legacy(self, key: str) -> None:
        """Warn when a legacy figure payload key is accessed."""
        target = _LEGACY_FIT_FIGURE_KEYS.get(key)
        if target is None:
            return
        warnings.warn(
            f"Accessing legacy figure payload key `{key}` is deprecated; use {target}.",
            DeprecationWarning,
            stacklevel=3,
        )


class FitResult(UserDict):
    """
    Generic fit result with payload, figures, status, and metadata.

    `data` stores the mapping-style fit payload. `figure` and `figures` hold
    the preferred visualization fields outside the payload, while `status`,
    `message`, and `created_at` store fit state and metadata.
    """

    status: FitStatus = FitStatus.SUCCESS

    def __init__(
        self,
        status: FitStatus,
        message: str | None = None,
        data: Mapping[str, Any] | None = None,
        *,
        figure: go.Figure | None = None,
        figures: Mapping[str, go.Figure] | None = None,
        created_at: str | None = None,
    ) -> None:
        """
        Initialize one fit result container.

        Parameters
        ----------
        status
            Outcome status of the fit operation.
        message
            Human-readable summary of the fit outcome.
        data
            Payload mapping returned by the fitting routine.
        figure
            Primary figure associated with the fit result.
        figures
            Named figure collection associated with the fit result.
        created_at
            Creation timestamp in ISO 8601 format. Defaults to the current UTC
            time when omitted.
        """
        super().__init__()
        self.data = _FitResultData(dict(data) if data is not None else {})
        self.status = status
        self.message = message
        self.figure = figure
        self.figures = figures
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        """Return a compact representation of the fit result."""
        return (
            f"<FitResult status={self.status.value} "
            f"message={self.message} data={{...}}>"
        )

    def get_figure(self, key: str | None = None) -> go.Figure:
        """
        Return one figure stored on the fit result.

        Parameters
        ----------
        key
            Name of the figure to return from `figures`. When omitted, this
            method returns the primary `figure`.

        Returns
        -------
        go.Figure
            Requested Plotly figure.

        Raises
        ------
        ValueError
            Raised when the primary figure is requested but `figure` is not set.
        KeyError
            Raised when a named figure is requested but `figures` does not
            contain the specified key.
        """
        if key is None:
            if self.figure is None:
                raise ValueError("FitResult does not contain a primary figure.")
            return self.figure

        figures = self.figures or {}
        try:
            return figures[key]
        except KeyError:
            raise KeyError(
                f"FitResult does not contain a figure named `{key}`."
            ) from None
