"""Directive-related protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol, TypeAlias


class DirectiveProtocol(Protocol):
    """Marker protocol for quelware directives."""


class CaptureModeProtocol(Protocol):
    """Minimal capture-mode enum value protocol."""

    @property
    def name(self) -> str:
        """Return capture mode name."""
        ...


CaptureModeValue: TypeAlias = CaptureModeProtocol | str


class CaptureModeNamespaceProtocol(Protocol):
    """Minimal capture-mode namespace protocol."""

    RAW_WAVEFORMS: CaptureModeValue
    AVERAGED_WAVEFORM: CaptureModeValue
    AVERAGED_VALUE: CaptureModeValue
    VALUES_PER_ITER: CaptureModeValue


class SetCaptureModeFactory(Protocol):
    """Factory protocol for `SetCaptureMode` directives."""

    def __call__(self, *, mode: CaptureModeValue) -> DirectiveProtocol:
        """Create one capture-mode directive."""
        ...
