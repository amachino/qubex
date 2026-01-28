"""Models for measurement scheduling."""

from __future__ import annotations

from qubex.core.model import Model
from qubex.pulse import PulseSchedule

from .capture_schedule import CaptureSchedule


class MeasurementSchedule(Model):
    """Pair of pulse and capture schedules for measurement."""

    pulse_schedule: PulseSchedule
    capture_schedule: CaptureSchedule
