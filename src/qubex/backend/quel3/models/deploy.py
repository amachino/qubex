"""Deployment models for QuEL-3 backend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoleName = Literal["TRANSMITTER", "TRANSCEIVER"]


@dataclass(frozen=True)
class InstrumentDeployRequest:
    """One QuEL-3 deploy request derived from logical target planning."""

    port_id: str
    role: RoleName
    frequency_range_min_hz: float
    frequency_range_max_hz: float
    alias: str
    target_labels: tuple[str, ...]
