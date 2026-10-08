"""Options and inspectable plans for QuEL-3 execution."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class Quel3ExecutionOptions(BaseModel):
    """
    Configure resource limits for packing adjacent jobs in input order.

    Sample limits count complex IQ samples, before averaging or decimation.
    Duration limits apply to each physical execution, using repeated timeline
    duration, not to the complete logical job. Compatible adjacent jobs are
    packed up to one-shot limits, then their shots are split to fit each run.
    Waveforms from different jobs are counted separately, even for equal IQ.
    Passing options to an execution replaces the controller's complete options.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    merge_jobs: bool = True
    max_waveform_samples: int = Field(default=65_536, gt=0)
    max_capture_samples: int = Field(default=15_000_000, gt=0)
    max_execution_duration_ns: float = Field(default=15_000_000_000, gt=0)


@dataclass(frozen=True)
class Quel3ResourceUsage:
    """Per-instrument waveform, per-MUX RAW capture, and repeated duration usage."""

    waveform_samples: dict[str, int]
    capture_samples: dict[str, int]
    duration_ns: float


@dataclass(frozen=True)
class Quel3PayloadPlacement:
    """Place one original payload at an offset in the execution timeline."""

    job_index: int
    start_offset_ns: float


@dataclass(frozen=True)
class Quel3PlannedExecution:
    """One physical trigger described by placements, shot range, and resource usage."""

    placements: tuple[Quel3PayloadPlacement, ...]
    shot_start: int
    shot_stop: int
    timeline_length_ns: float
    resources: Quel3ResourceUsage


@dataclass(frozen=True)
class Quel3ExecutionPlan:
    """
    Inspect a complete batch plan without opening hardware sessions.

    Placements use zero-based input indices. Plans contain resource metadata
    and scheduling decisions without waveform buffers or hardware payloads.
    Estimated duration sums repeated timeline lengths, including shot intervals.
    It excludes sequencer alignment, quelware's iteration blank, and hardware
    setup, transfer, and trigger overhead.
    """

    payload_count: int
    executions: tuple[Quel3PlannedExecution, ...]

    @property
    def estimated_duration_ns(self) -> float:
        """Return the summed duration of physical execution timelines in ns."""
        return sum(run.resources.duration_ns for run in self.executions)
