"""DC voltage state model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DCVoltageState:
    """Readback state for one mux-connected DC output."""

    mux_label: str
    mux_index: int
    channel: int
    voltage: float
    output: Literal["on", "off"]

    @property
    def is_on(self) -> bool:
        """Return whether the DC output is on."""
        return self.output == "on"
