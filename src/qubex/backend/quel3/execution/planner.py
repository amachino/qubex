"""Plan adjacent payload groups and shot ranges from resource metadata."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from qubex.backend.quel3.models.execution_plan import (
    Quel3ExecutionOptions,
    Quel3ExecutionPlan,
    Quel3PayloadPlacement,
    Quel3PlannedExecution,
    Quel3ResourceUsage,
)

from .resources import ExecutionConditions, PayloadPlanningInfo, ResourceRequirements
from .timing import ceil_to_grid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PayloadGroup:
    placements: tuple[Quel3PayloadPlacement, ...]
    requirements: ResourceRequirements
    conditions: ExecutionConditions

    @property
    def shot_duration_ns(self) -> float:
        return self.requirements.timeline_duration_ns + self.conditions.shot_interval_ns


class Quel3ExecutionPlanner:
    """Pack whole adjacent jobs and split only oversized individual jobs."""

    def plan(
        self,
        payloads: tuple[PayloadPlanningInfo, ...],
        *,
        options: Quel3ExecutionOptions,
    ) -> Quel3ExecutionPlan:
        """Determine placements and shot ranges without constructing hardware data."""
        groups = self._group_adjacent_payloads(payloads, options)
        executions = tuple(
            execution
            for group in groups
            for execution in self._split_shots(group, options)
        )
        return Quel3ExecutionPlan(len(payloads), executions)

    def _group_adjacent_payloads(
        self,
        payloads: tuple[PayloadPlanningInfo, ...],
        options: Quel3ExecutionOptions,
    ) -> list[_PayloadGroup]:
        groups: list[_PayloadGroup] = []
        current: _PayloadGroup | None = None
        for index in range(len(payloads)):
            single = self._make_group(payloads, (index,))
            violations = self._resource_violations(single, options)
            if violations:
                raise ValueError(
                    f"QuEL-3 job {index} exceeds a one-shot resource limit: {'; '.join(violations)}"
                )
            indices = (
                tuple(p.job_index for p in current.placements)
                if current is not None
                else ()
            )
            candidate = (*indices, index)
            if indices:
                reason = (
                    self._incompatibility_reason(payloads, candidate)
                    if options.merge_jobs
                    else "merge_jobs=False"
                )
                if reason is None:
                    merged = self._make_group(payloads, candidate)
                    if (
                        not self._resource_violations(merged, options)
                        and self._max_shots_per_execution(merged, options)
                        == merged.conditions.n_iterations
                    ):
                        current = merged
                        continue
                    reason = "; ".join(
                        self._resource_violations(
                            merged, options, shots=merged.conditions.n_iterations
                        )
                    )
                logger.debug(
                    "QuEL-3 packing boundary before_job=%d group_jobs=%s reason=%s",
                    index,
                    indices,
                    reason,
                )
            if current is not None:
                groups.append(current)
            current = single
        if current is not None:
            groups.append(current)
        return groups

    @staticmethod
    def _incompatibility_reason(
        payloads: tuple[PayloadPlanningInfo, ...], indices: tuple[int, ...]
    ) -> str | None:
        first = payloads[indices[0]].conditions
        frequencies: dict[str, float] = {}
        capture_periods: set[int] = set()
        for index in indices:
            conditions = payloads[index].conditions
            for setting, value, reference in (
                ("n_iterations", conditions.n_iterations, first.n_iterations),
                ("capture_mode", conditions.capture_mode, first.capture_mode),
                (
                    "shot_interval_ns",
                    conditions.shot_interval_ns,
                    first.shot_interval_ns,
                ),
                ("cable_delay_ns", conditions.cable_delay_ns, first.cable_delay_ns),
            ):
                if value != reference:
                    return f"{setting}: job={index} value={value} current={reference}"
            if conditions.capture_sampling_period_fs is not None:
                capture_periods.add(conditions.capture_sampling_period_fs)
            for alias, frequency in conditions.frequencies_hz.items():
                if alias in frequencies and frequencies[alias] != frequency:
                    return f"frequency {alias}: job={index} value={frequency} current={frequencies[alias]} Hz"
                frequencies[alias] = frequency
        if len(capture_periods) > 1:
            return f"capture_sampling_period_fs: values={sorted(capture_periods)}"
        return None

    @staticmethod
    def _make_group(
        payloads: tuple[PayloadPlanningInfo, ...], indices: tuple[int, ...]
    ) -> _PayloadGroup:
        conditions = payloads[indices[0]].conditions
        grid_fs = math.lcm(
            *(payloads[index].requirements.sampling_grid_fs for index in indices)
        )
        waveform_samples: dict[str, int] = {}
        captures: dict[str, int] = {}
        placements = []
        end_ns = 0.0
        for position, index in enumerate(indices):
            if position:
                end_ns = ceil_to_grid(end_ns + conditions.shot_interval_ns, grid_fs)
            placements.append(Quel3PayloadPlacement(index, end_ns))
            requirements = payloads[index].requirements
            end_ns += requirements.timeline_duration_ns
            for alias, count in requirements.waveform_samples.items():
                waveform_samples[alias] = waveform_samples.get(alias, 0) + count
            for receiver, count in requirements.capture_samples_per_shot.items():
                captures[receiver] = captures.get(receiver, 0) + count
        return _PayloadGroup(
            tuple(placements),
            ResourceRequirements(waveform_samples, captures, end_ns, grid_fs),
            conditions,
        )

    @staticmethod
    def _resource_violations(
        group: _PayloadGroup, options: Quel3ExecutionOptions, *, shots: int = 1
    ) -> list[str]:
        requirements = group.requirements
        violations = [
            f"waveform {alias}: required={count} limit={options.max_waveform_samples}"
            for alias, count in requirements.waveform_samples.items()
            if count > options.max_waveform_samples
        ]
        violations.extend(
            f"capture {receiver}: required={count * shots} limit={options.max_capture_samples}"
            for receiver, count in requirements.capture_samples_per_shot.items()
            if count * shots > options.max_capture_samples
        )
        duration_ns = group.shot_duration_ns * shots
        if duration_ns > options.max_execution_duration_ns:
            violations.append(
                f"duration: required={duration_ns} limit={options.max_execution_duration_ns} ns"
            )
        return violations

    @staticmethod
    def _max_shots_per_execution(
        group: _PayloadGroup, options: Quel3ExecutionOptions
    ) -> int:
        requirements = group.requirements
        capture_limits = [
            options.max_capture_samples // count
            for count in requirements.capture_samples_per_shot.values()
        ]
        duration_limit = options.max_execution_duration_ns / group.shot_duration_ns
        rounded_limit = round(duration_limit)
        # Keep exact inclusive boundaries despite floating-point division.
        duration_shots = (
            rounded_limit
            if math.isclose(duration_limit, rounded_limit, rel_tol=0, abs_tol=1e-9)
            else math.floor(duration_limit)
        )
        return min(group.conditions.n_iterations, duration_shots, *capture_limits)

    def _split_shots(
        self, group: _PayloadGroup, options: Quel3ExecutionOptions
    ) -> list[Quel3PlannedExecution]:
        requirements = group.requirements
        shot_limit = self._max_shots_per_execution(group, options)
        if shot_limit < group.conditions.n_iterations:
            logger.debug(
                "QuEL-3 shot split jobs=%s shots=%d max_shots_per_execution=%d reason=%s",
                tuple(placement.job_index for placement in group.placements),
                group.conditions.n_iterations,
                shot_limit,
                "; ".join(
                    self._resource_violations(
                        group, options, shots=group.conditions.n_iterations
                    )
                ),
            )
        executions = []
        for start in range(0, group.conditions.n_iterations, shot_limit):
            stop = min(start + shot_limit, group.conditions.n_iterations)
            shots = stop - start
            executions.append(
                Quel3PlannedExecution(
                    placements=group.placements,
                    shot_start=start,
                    shot_stop=stop,
                    timeline_length_ns=requirements.timeline_duration_ns,
                    resources=Quel3ResourceUsage(
                        requirements.waveform_samples,
                        {
                            receiver: count * shots
                            for receiver, count in requirements.capture_samples_per_shot.items()
                        },
                        group.shot_duration_ns * shots,
                    ),
                )
            )
        return executions
