"""Directive-related protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol


class DirectiveProtocol(Protocol):
    """Marker protocol for quelware directives."""


class SetCaptureModeFactory(Protocol):
    """Factory protocol for `SetCaptureMode` directives."""

    def __call__(self, *, mode: object) -> DirectiveProtocol:
        """Create one capture-mode directive."""
        ...
