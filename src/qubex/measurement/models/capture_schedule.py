from __future__ import annotations

from functools import cached_property

from qubex.core.model import Model


class Capture(Model):
    channels: list[str]
    start_time: float
    duration: float


class CaptureSchedule(Model):
    captures: list[Capture]

    @cached_property
    def channels(self) -> dict[str, list[Capture]]:
        schedule: dict[str, list[Capture]] = {}
        for capture in self.captures:
            for channel in capture.channels:
                if channel not in schedule:
                    schedule[channel] = []
                schedule[channel].append(capture)
        return schedule
