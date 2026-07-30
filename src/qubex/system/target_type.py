"""Target-type enumeration shared across system models."""

from __future__ import annotations

from enum import Enum


class TargetType(Enum):
    """Enumerate supported logical target types."""

    CTRL_GE = "CTRL_GE"
    CTRL_EF = "CTRL_EF"
    CTRL_FH = "CTRL_FH"
    CTRL_2Q = "CTRL_2Q"
    CTRL_CR = "CTRL_CR"
    READ = "READ"
    PUMP = "PUMP"
    UNKNOWN = "UNKNOWN"
