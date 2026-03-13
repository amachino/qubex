"""Lightweight result container for experiments."""

from __future__ import annotations

import warnings
from collections import UserDict
from collections.abc import Mapping
from datetime import datetime, timezone

from plotly.graph_objects import Figure

_LEGACY_FIGURE_KEYS = {
    "fig": "the `figure` attribute",
    "figure": "the `figure` attribute",
    "figs": "the `figures` attribute",
    "figures": "the `figures` attribute",
}


class _ResultData(dict[str, object]):
    """
    Payload mapping for `Result`.

    This wrapper preserves plain mapping behavior while warning when callers
    read legacy figure keys such as `"fig"` and `"figures"`.
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
        target = _LEGACY_FIGURE_KEYS.get(key)
        if target is None:
            return
        warnings.warn(
            f"Accessing legacy figure payload key `{key}` is deprecated; use `{target}`.",
            DeprecationWarning,
            stacklevel=3,
        )


class Result(UserDict):
    """
    Generic experiment result with payload, figures, and metadata.

    `data` stores the mapping-style payload exposed by existing experiment APIs.
    `figure` and `figures` hold the preferred visualization fields outside the
    payload, while `created_at` records when the result was created.
    """

    def __init__(
        self,
        data: Mapping[str, object] | None = None,
        *,
        figure: Figure | None = None,
        figures: Mapping[str, Figure] | None = None,
        created_at: str | None = None,
    ) -> None:
        """
        Initialize an experiment result container.

        Parameters
        ----------
        data
            Payload mapping returned by the experiment API.
        figure
            Primary figure associated with the result.
        figures
            Named figure collection associated with the result.
        created_at
            Creation timestamp in ISO 8601 format. Defaults to the current UTC
            time when omitted.
        """
        super().__init__()
        self.data = _ResultData(dict(data) if data is not None else {})
        self.figure = figure
        self.figures = figures
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"<Result created_at={self.created_at} data={{...}}>"
