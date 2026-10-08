"""Restore logical results from split and merged QuEL-3 executions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qubex.backend.quel3.models.payload import Quel3CaptureMode, Quel3ExecutionPayload
from qubex.backend.quel3.models.result import Quel3BackendExecutionResult

from .payload_builder import PayloadResultFragment, ResultMapping
from .timing import sample_count


@dataclass
class _PayloadResultAccumulator:
    """Collect contiguous shot chunks and restore one original payload."""

    payload: Quel3ExecutionPayload
    job_index: int
    shots_received: int = 0
    config: dict[str, float] = field(default_factory=dict)
    status: dict[str, object] = field(default_factory=dict)
    shot_counts: list[int] = field(default_factory=list)
    capture_chunks: dict[tuple[str, int], list[np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        self.capture_chunks = {
            (alias, index): []
            for alias, timeline in self.payload.fixed_timelines.items()
            for index in range(len(timeline.capture_windows))
        }

    def add(
        self,
        fragment: PayloadResultFragment,
        result: Quel3BackendExecutionResult,
        sampling_period_ns: float,
    ) -> None:
        """Validate and copy the next shot chunk for this payload."""
        if self.shots_received != fragment.shot_start:
            raise ValueError(
                f"Unexpected or duplicate shot range for job {self.job_index}."
            )
        if self.shots_received and self.config != result.config:
            raise ValueError("Sampling metadata changed between shot chunks.")
        shots = fragment.shot_stop - fragment.shot_start
        for capture in fragment.captures:
            data = np.asarray(
                result.data[capture.alias][capture.execution_index], dtype=np.complex128
            )
            expected = self._capture_shape(
                capture.alias, capture.original_index, shots, sampling_period_ns
            )
            if data.shape != expected:
                raise ValueError(
                    f"Invalid capture shape or shot count for job {self.job_index}, {capture.alias}: expected {expected}, got {data.shape}."
                )
            self.capture_chunks[capture.alias, capture.original_index].append(
                data.copy()
            )
        self.shots_received = fragment.shot_stop
        self.shot_counts.append(shots)
        self.config = dict(result.config)
        self.status.update(result.status)

    def _capture_shape(
        self, alias: str, index: int, shots: int, period_ns: float
    ) -> tuple[int, ...]:
        window = self.payload.fixed_timelines[alias].capture_windows[index]
        samples = max(1, sample_count(window.length_ns, period_ns))
        return {
            Quel3CaptureMode.RAW_WAVEFORMS: (shots, samples),
            Quel3CaptureMode.VALUES_PER_ITER: (shots,),
            Quel3CaptureMode.AVERAGED_WAVEFORM: (samples,),
            Quel3CaptureMode.AVERAGED_VALUE: (1,),
        }[self.payload.capture_mode]

    def finish(self) -> Quel3BackendExecutionResult:
        """Concatenate shots or weight averages by the number of shots per chunk."""
        if self.shots_received != self.payload.n_iterations:
            raise ValueError("Incomplete shot coverage in execution results.")
        averaged = self.payload.capture_mode in (
            Quel3CaptureMode.AVERAGED_VALUE,
            Quel3CaptureMode.AVERAGED_WAVEFORM,
        )
        data: dict[str, list[np.ndarray]] = {}
        for (alias, _), chunks in self.capture_chunks.items():
            samples = (
                np.average(chunks, axis=0, weights=self.shot_counts)
                if averaged
                else np.concatenate(chunks, axis=0)
            )
            data.setdefault(alias, []).append(samples)
        return Quel3BackendExecutionResult(self.status, data, self.config)


class Quel3ResultAssembler:
    """Check physical result structure and route captures to payload accumulators."""

    def __init__(self, payloads: tuple[Quel3ExecutionPayload, ...]) -> None:
        self._accumulators = tuple(
            _PayloadResultAccumulator(payload, index)
            for index, payload in enumerate(payloads)
        )

    def add(self, mapping: ResultMapping, result: Quel3BackendExecutionResult) -> None:
        """Accept one successful physical execution exactly once."""
        period = result.config.get("sampling_period_ns")
        if period is None or not np.isfinite(period) or period <= 0:
            raise ValueError("Result sampling_period_ns must be finite and positive.")
        if set(result.data) - set(mapping.capture_counts):
            raise ValueError("Result contains unexpected instrument aliases.")
        for alias, expected_count in mapping.capture_counts.items():
            count = len(result.data[alias]) if alias in result.data else 0
            if count != expected_count:
                raise ValueError(f"Missing or unexpected captures for {alias!r}.")
        for fragment in mapping.fragments:
            self._accumulators[fragment.job_index].add(fragment, result, period)

    def finish(self) -> list[Quel3BackendExecutionResult]:
        """Return complete logical results in the original payload order."""
        return [accumulator.finish() for accumulator in self._accumulators]
