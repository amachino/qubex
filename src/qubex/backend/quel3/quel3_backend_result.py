"""Execution-result payload for QuEL-3 backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubex.typing import MeasurementMode


@dataclass
class Quel3BackendExecutionResult:
    """Backend-level measurement result returned by QuEL-3 execution."""

    mode: MeasurementMode
    data: dict[str, list[np.ndarray]]
    sampling_period_ns: float | None = None
