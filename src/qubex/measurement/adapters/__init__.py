"""Measurement backend adapters."""

# Backward-compatible re-export: canonical ownership is `qubex.backend.quel3`.
from qubex.backend.quel3 import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
    Quel3WaveformDefinition,
    Quel3WaveformEvent,
)

from .backend_adapter import (
    MeasurementBackendAdapter,
    Quel1MeasurementBackendAdapter,
    Quel3MeasurementBackendAdapter,
)

__all__ = [
    "MeasurementBackendAdapter",
    "Quel1MeasurementBackendAdapter",
    "Quel3CaptureWindow",
    "Quel3ExecutionPayload",
    "Quel3MeasurementBackendAdapter",
    "Quel3TargetTimeline",
    "Quel3WaveformDefinition",
    "Quel3WaveformEvent",
]
