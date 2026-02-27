"""Execution-result payload for QuEL-3 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Quel3BackendExecutionResult:
    """Backend-level measurement result returned by QuEL-3 execution."""

    status: dict[str, Any]
    data: dict[str, list[np.ndarray]]
    config: dict[str, Any]
