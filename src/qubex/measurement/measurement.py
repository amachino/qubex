"""Backward-compatible Measurement module alias."""

from typing_extensions import deprecated

from .measurement_client import MeasurementClient


@deprecated("Measurement is deprecated, use MeasurementClient instead.")
class Measurement(MeasurementClient):
    """Deprecated alias for `MeasurementClient`."""


__all__ = ["Measurement"]
