"""Plan adjacent payload groups and shot ranges from resource metadata."""

from __future__ import annotations

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


@dataclass(frozen=True)
class _PayloadGroup:
    placements: tuple[Quel3PayloadPlacement, ...]
    requirements: ResourceRequirements
    conditions: ExecutionConditions

    @property
    def shot_duration_ns(self) -> float:
        return self.requirements.timeline_duration_ns + self.conditions.shot_interval_ns


class Quel3ExecutionPlanner:
    """Group compatible neighbors, then split shots within each resource budget."""

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
            violations = self._one_shot_violations(single, options)
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
            if indices and options.merge_jobs and self._compatible(payloads, candidate):
                merged = self._make_group(payloads, candidate)
                if not self._one_shot_violations(merged, options):
                    current = merged
                    continue
            if current is not None:
                groups.append(current)
            current = single
        if current is not None:
            groups.append(current)
        return groups

    @staticmethod
    def _compatible(
        payloads: tuple[PayloadPlanningInfo, ...], indices: tuple[int, ...]
    ) -> bool:
        first = payloads[indices[0]].conditions
        frequencies: dict[str, float] = {}
        capture_periods: set[int] = set()
        for index in indices:
            conditions = payloads[index].conditions
            if (
                conditions.n_iterations,
                conditions.capture_mode,
                conditions.shot_interval_ns,
                conditions.cable_delay_ns,
            ) != (
                first.n_iterations,
                first.capture_mode,
                first.shot_interval_ns,
                first.cable_delay_ns,
            ):
                return False
            if conditions.capture_sampling_period_fs is not None:
                capture_periods.add(conditions.capture_sampling_period_fs)
            for alias, frequency in conditions.frequencies_hz.items():
                if frequency is None or (
                    alias in frequencies and frequencies[alias] != frequency
                ):
                    return False
                frequencies[alias] = frequency
        return len(capture_periods) <= 1

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
    def _one_shot_violations(
        group: _PayloadGroup, options: Quel3ExecutionOptions
    ) -> list[str]:
        requirements = group.requirements
        violations = [
            f"waveform {alias}: required={count} limit={options.max_waveform_samples}"
            for alias, count in requirements.waveform_samples.items()
            if count > options.max_waveform_samples
        ]
        violations.extend(
            f"capture {receiver}: required={count} limit={options.max_capture_samples}"
            for receiver, count in requirements.capture_samples_per_shot.items()
            if count > options.max_capture_samples
        )
        if group.shot_duration_ns > options.max_execution_duration_ns:
            violations.append(
                f"duration: required={group.shot_duration_ns} limit={options.max_execution_duration_ns} ns"
            )
        return violations

    @staticmethod
    def _split_shots(
        group: _PayloadGroup, options: Quel3ExecutionOptions
    ) -> list[Quel3PlannedExecution]:
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
        shot_limit = min(group.conditions.n_iterations, duration_shots, *capture_limits)
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
