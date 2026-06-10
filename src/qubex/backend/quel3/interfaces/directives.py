"""Directive-related protocols for QuEL-3 quelware integration."""

from __future__ import annotations

from typing import Protocol


class DirectiveProtocol(Protocol):
    """Marker protocol for quelware directives."""


class CaptureModeProtocol(Protocol):
    """Capture-mode enum value protocol."""

    @property
    def name(self) -> str:
        """Return capture mode name."""
        ...

    @property
    def value(self) -> int:
        """Return capture mode value."""
        ...


class CaptureModeNamespaceProtocol(Protocol):
    """Capture-mode enum namespace protocol."""

    RAW_WAVEFORMS: CaptureModeProtocol
    AVERAGED_WAVEFORM: CaptureModeProtocol
    AVERAGED_VALUE: CaptureModeProtocol
    VALUES_PER_ITER: CaptureModeProtocol


class SetCaptureModeFactory(Protocol):
    """Factory protocol for `SetCaptureMode` directives."""

    def __call__(self, *, mode: CaptureModeProtocol) -> DirectiveProtocol:
        """Create one capture-mode directive."""
        ...


class SetFrequencyFactory(Protocol):
    """Factory protocol for `SetFrequency` directives."""

    def __call__(self, *, hz: float) -> DirectiveProtocol:
        """Create one frequency directive."""
        ...
