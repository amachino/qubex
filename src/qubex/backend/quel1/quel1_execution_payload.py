"""QuEL-1 backend execution payload models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Quel1ExecutionPayload:
    """QuEL-1 execution payload carrying sequencer compilation inputs."""

    gen_sampled_sequence: dict[str, Any]
    cap_sampled_sequence: dict[str, Any]
    resource_map: dict[str, list[dict[str, Any]]]
    interval: int
    repeats: int
    integral_mode: str
    dsp_demodulation: bool
    enable_sum: bool
    enable_classification: bool
    line_param0: tuple[float, float, float]
    line_param1: tuple[float, float, float]
