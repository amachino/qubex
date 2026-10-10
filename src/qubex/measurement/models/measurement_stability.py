"""Models for measurement stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

OutputSignalKind = Literal["control", "readout"]
OutputSignalReferenceScope = Literal["box", "target"]


@dataclass(frozen=True)
class MonitorStatistic:
    """Amplitude and phase statistics for one monitor-path capture."""

    reference_target: str
    covered_targets: tuple[str, ...]
    monitor_target: str
    capture_index: int
    amplitude_mean: float
    amplitude_std: float
    amplitude_sem: float
    amplitude_relative_sem: float
    phase_mean_rad: float
    phase_std_rad: float
    phase_sem_rad: float
    phase_resultant_length: float
    n_shots: int
    n_samples: int


@dataclass(frozen=True)
class OutputSignalCorrection:
    """Session-local output gain and phase correction for one generator target."""

    target: str
    kind: OutputSignalKind
    port_id: str
    monitor_target: str
    reference_amplitude: float
    reference_amplitude_sem: float
    measured_amplitude: float
    measured_amplitude_sem: float
    reference_phase_rad: float
    reference_phase_sem_rad: float
    measured_phase_rad: float
    measured_phase_sem_rad: float
    gain: float
    previous_gain: float
    raw_gain: float
    limited_gain: float
    effective_gain_correction_deadband: float
    phase_offset_rad: float
    previous_phase_offset_rad: float
    raw_phase_offset_rad: float
    limited_phase_offset_rad: float
    effective_phase_correction_deadband: float
    reference_target: str | None = None
    reference_scope: OutputSignalReferenceScope = "target"


@dataclass(frozen=True)
class MeasurementStabilitySnapshot:
    """Snapshot of session-local measurement stability state."""

    output_corrections: dict[str, OutputSignalCorrection]
    signals: dict[str, MonitorStatistic] = field(default_factory=dict)
    sample_index: int | None = None
    elapsed_s: float | None = None
    timestamp: str | None = None
