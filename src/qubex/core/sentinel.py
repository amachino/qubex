"""Shared sentinel utilities."""

from __future__ import annotations

from typing_extensions import Sentinel


def make_sentinel(name: str) -> object:
    """Create one named sentinel object."""
    return Sentinel(name)


MISSING = make_sentinel("MISSING")
