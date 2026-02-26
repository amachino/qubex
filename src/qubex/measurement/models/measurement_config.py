"""Measurement configuration model."""

from __future__ import annotations

from qubex.core import Model


class MeasurementConfig(Model):
    """Measurement configuration model."""

    n_shots: int
    shot_interval: float
    shot_averaging: bool
    time_integration: bool
    state_classification: bool
