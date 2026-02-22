"""Backward-compatible MeasurementClient alias module."""

from __future__ import annotations

from typing_extensions import deprecated

from .measurement import Measurement


@deprecated("MeasurementClient is deprecated, use Measurement instead.")
class MeasurementClient(Measurement):
    """Deprecated alias for `Measurement`."""


__all__ = ["MeasurementClient"]
