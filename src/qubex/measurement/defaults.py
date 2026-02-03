"""Backward-compatible exports for measurement defaults."""

from __future__ import annotations

from .measurement_defaults import (
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMPTIME,
    DEFAULT_SHOTS,
)

__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_READOUT_DURATION",
    "DEFAULT_READOUT_POST_MARGIN",
    "DEFAULT_READOUT_PRE_MARGIN",
    "DEFAULT_READOUT_RAMPTIME",
    "DEFAULT_SHOTS",
]
