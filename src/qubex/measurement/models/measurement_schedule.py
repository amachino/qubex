from __future__ import annotations

from qubex.core.model import Model
from qubex.pulse import PulseSchedule

from .capture_schedule import CaptureSchedule


class MeasurementSchedule(Model):
    pulse_schedule: PulseSchedule
    capture_schedule: CaptureSchedule
