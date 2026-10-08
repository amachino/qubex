"""Build hardware payloads and result mappings from planned placements."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from qubex.backend.quel3.models.execution_plan import (
    Quel3ExecutionPlan,
    Quel3PlannedExecution,
    Quel3ResourceUsage,
)
from qubex.backend.quel3.models.payload import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3WaveformEvent,
)

from .resources import PayloadAnalysis


@dataclass(frozen=True)
class CaptureMapping:
    """Map one original capture to its index in the physical result."""

    alias: str
    original_index: int
    execution_index: int


@dataclass(frozen=True)
class PayloadResultFragment:
    """Original job and half-open shot range represented by one execution."""

    job_index: int
    shot_start: int
    shot_stop: int
    captures: tuple[CaptureMapping, ...]


@dataclass(frozen=True)
class ResultMapping:
    """Expected capture counts and routes back to original payload captures."""

    capture_counts: dict[str, int]
    fragments: tuple[PayloadResultFragment, ...]


@dataclass(frozen=True)
class BuiltExecution:
    """Hardware payload, result routing, and resource usage for one trigger."""

    payload: Quel3ExecutionPayload
    result_mapping: ResultMapping
    resources: Quel3ResourceUsage


@dataclass
class _TimelineBuilder:
    frequency_hz: float | None
    events: list[Quel3WaveformEvent] = field(default_factory=list)
    windows: list[Quel3CaptureWindow] = field(default_factory=list)

    def append(
        self,
        timeline: Quel3FixedTimeline,
        *,
        alias: str,
        job_index: int,
        offset_ns: float,
        waveform_names: dict[str, str],
        namespace: bool,
    ) -> tuple[CaptureMapping, ...]:
        mappings = []
        for event in timeline.events:
            self.events.append(
                replace(
                    event,
                    waveform_name=waveform_names[event.waveform_name],
                    start_offset_ns=event.start_offset_ns + offset_ns,
                )
            )
        for index, window in enumerate(timeline.capture_windows):
            mappings.append(CaptureMapping(alias, index, len(self.windows)))
            self.windows.append(
                replace(
                    window,
                    name=f"job{job_index}_capture{index}" if namespace else window.name,
                    start_offset_ns=window.start_offset_ns + offset_ns,
                )
            )
        return tuple(mappings)

    def finish(self, length_ns: float) -> Quel3FixedTimeline:
        return Quel3FixedTimeline(
            tuple(self.events), tuple(self.windows), length_ns, self.frequency_hz
        )


class Quel3ExecutionPayloadBuilder:
    """Apply plan decisions without checking budgets or recalculating placements."""

    def build_all(
        self, analyses: tuple[PayloadAnalysis, ...], plan: Quel3ExecutionPlan
    ) -> tuple[BuiltExecution, ...]:
        """Construct each planned hardware payload and its result mapping."""
        return tuple(self.build(analyses, execution) for execution in plan.executions)

    def build(
        self, analyses: tuple[PayloadAnalysis, ...], execution: Quel3PlannedExecution
    ) -> BuiltExecution:
        """Namespace and shift payload content according to its planned offsets."""
        first = analyses[execution.placements[0].job_index].payload
        namespace = len(execution.placements) > 1
        library = {}
        timelines: dict[str, _TimelineBuilder] = {}
        fragments = []
        for placement in execution.placements:
            index = placement.job_index
            payload = analyses[index].payload
            names = {
                name: f"job{index}_{name}" if namespace else name
                for name in payload.waveform_library
            }
            library.update(
                (names[name], waveform)
                for name, waveform in payload.waveform_library.items()
            )
            mappings = []
            for alias, timeline in payload.fixed_timelines.items():
                if alias not in timelines:
                    timelines[alias] = _TimelineBuilder(timeline.frequency_hz)
                captures = timelines[alias].append(
                    timeline,
                    alias=alias,
                    job_index=index,
                    offset_ns=placement.start_offset_ns,
                    waveform_names=names,
                    namespace=namespace,
                )
                mappings.extend(captures)
            fragments.append(
                PayloadResultFragment(
                    index, execution.shot_start, execution.shot_stop, tuple(mappings)
                )
            )
        payload = replace(
            first,
            waveform_library=library,
            fixed_timelines={
                alias: builder.finish(execution.timeline_length_ns)
                for alias, builder in timelines.items()
            },
            n_iterations=execution.shot_stop - execution.shot_start,
        )
        mapping = ResultMapping(
            {alias: len(builder.windows) for alias, builder in timelines.items()},
            tuple(fragments),
        )
        return BuiltExecution(payload, mapping, execution.resources)
