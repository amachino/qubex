from __future__ import annotations

"""Deprecated: use `qubex.errors` instead.

This module re-exports exceptions for backward compatibility.
"""

from ..errors import BackendUnavailableError, CalibrationMissingError

__all__ = [
    "CalibrationMissingError",
    "BackendUnavailableError",
]
