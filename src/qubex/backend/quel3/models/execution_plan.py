"""Options and inspectable plans for QuEL-3 execution."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Quel3ExecutionOptions(BaseModel):
    """
    Configure resource limits for packing adjacent jobs in input order.

    Hardware sample limits count complex IQ samples before averaging or decimation.
    The return limit counts all returned IQ samples across instruments and units,
    after capture-mode reduction.
    Duration limits apply to each physical execution, using repeated timeline
    duration, not to the complete logical job. Compatible adjacent jobs are
    packed only when all requested shots fit in one run. Shots are split only
    for individual jobs that cannot execute all requested shots in one run.
    Waveforms from different jobs are counted separately, even for equal IQ.
    `compress_waveforms` defaults to True and reduces verified Rect/FlatTop
    constant-envelope regions on control and readout channels before planning.
    Set it to False to preserve uncompressed registration. Configure it under
    `quel3.execution` in `system.yaml` or override the complete execution options.
    `compression_target_block_samples` defaults to 5000 registered samples;
    `compression_min_chunk_samples` defaults to 1000 and bounds every new
    fragment, including prefixes and tails. Both must be positive integers,
    with the minimum no greater than the target. Short original waves are kept.
    Memory feasibility is checked by the planner after compression.
    Passing options to an execution replaces the controller's complete options.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )

    merge_jobs: bool = True
    compress_waveforms: bool = True
    compression_target_block_samples: int = Field(default=5000, gt=0)
    compression_min_chunk_samples: int = Field(default=1000, gt=0)
    max_waveform_samples: int = Field(default=60_000, gt=0)
    max_capture_samples: int = Field(default=15_000_000, gt=0)
    max_return_capture_samples: int = Field(default=3_125_000, gt=0)
    max_execution_duration_ns: float = Field(default=15_000_000_000, gt=0)

    @model_validator(mode="after")
    def _validate_compression_lengths(self) -> Quel3ExecutionOptions:
        """Require the minimum fragment length to fit within the target block."""
        if self.compression_min_chunk_samples > self.compression_target_block_samples:
            raise ValueError(
                "Compression minimum chunk length must not exceed target block length."
            )
        return self


@dataclass(frozen=True)
class Quel3ResourceUsage:
    """Waveform, RAW capture, returned capture, and duration usage for one trigger."""

    waveform_samples: dict[str, int]
    capture_samples: dict[str, int]
    duration_ns: float
    return_capture_samples: int = 0


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
