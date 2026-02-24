"""Execution-result payload for QuEL-3 backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qubex.typing import MeasurementMode


@dataclass
class Quel3BackendExecutionResult:
    """Backend-level measurement result returned by QuEL-3 execution."""

    mode: MeasurementMode
    data: dict[str, list[np.ndarray]]
    device_config: dict[str, Any] = field(default_factory=dict)
    measurement_config: dict[str, Any] = field(default_factory=dict)
    sampling_period_ns: float | None = None
    avg_sample_stride: int | None = None
