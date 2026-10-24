"""Capture schedule models."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

from qubex.core import Model

CapturePlacement = Literal["pulse_aligned", "entire_schedule", "explicit"]
"""How capture windows were placed.

`pulse_aligned` derives one window per readout pulse, `entire_schedule` spans the
whole schedule with a single window, and `explicit` carries windows positioned by
the caller independently of the readout pulses.
"""


class Capture(Model):
    """Single capture window definition."""

    channels: list[str]
    start_time: float
    duration: float
    is_workaround: bool = False


class CaptureSchedule(Model):
    """Collection of capture windows."""

    captures: list[Capture]
    placement: CapturePlacement = "pulse_aligned"

    @cached_property
    def channels(self) -> dict[str, list[Capture]]:
        """Return captures grouped by channel."""
        schedule: dict[str, list[Capture]] = {}
        for capture in self.captures:
            for channel in capture.channels:
                if channel not in schedule:
                    schedule[channel] = []
                schedule[channel].append(capture)
        return schedule
