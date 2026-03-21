"""Execution-result payload for QuEL-3 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Quel3BackendResultConfig: TypeAlias = dict[str, float]


@dataclass
class Quel3BackendExecutionResult:
    """Backend-level measurement result returned by QuEL-3 execution."""

    status: dict[str, object]
    data: dict[str, list[np.ndarray]]
    config: Quel3BackendResultConfig
