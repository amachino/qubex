# Pulse Copy Depth Policy

## Status

- State: `IMPLEMENTED`
- Documented on: 2026-02-22

## Purpose

Define when pulse-related copy operations should use shallow copy versus deep copy.
This note is focused on performance-sensitive sampling paths in `qxpulse`.

## Decision Rules

Use `shallow` copy when all of the following are true:

- The copied object is used only as a temporary view during evaluation/sampling.
- Only scalar waveform modifiers are changed (`_scale`, `_phase`, `_detuning`).
- The original structure (`_elements`, schedule/channel containers) is not mutated.

Use `deep` copy when any of the following are true:

- The API contract promises an independent object graph to callers.
- Container members may be mutated after copy (`_elements`, channels, schedule internals).
- The operation returns a user-visible object that must be safe to edit in isolation.

## Current Policy by Type

`Pulse`:

- `Pulse.__copy__`: `shallow` + materialize once.
- `Pulse.__deepcopy__`: `deep` + materialize once.
- `Pulse.copy()`: uses `deepcopy(self)` (public detached copy contract).
- `Pulse.scaled/detuned/shifted`: use `shallow` copy (`copy.copy(self)`), then update scalar modifiers.

`PulseArray`:

- `PulseArray.flattened_elements`: `copy.copy(obj)` (temporary flatten/evaluation path).
- `PulseArray.copy()`: `copy.deepcopy(self)` (detached container copy).
- `PulseArray.scaled/detuned/shifted/repeated/added`: remain `deep` (public object-returning transforms on mutable container structure).

`PulseSchedule`:

- `PulseSchedule.copy()` and schedule-level transforms remain `deep`.
- Rationale: schedule/channel structures are mutable and user-visible.

## Why This Split

- `flattened_elements` is on the hot path of sampling and should avoid deep-copying sampled arrays.
- Deep copy in hot loops causes avoidable CPU and memory overhead.
- Public copy/transform APIs still need strong isolation guarantees for callers.

## Anti-Patterns

- Do not replace all copy sites with `deepcopy` "for safety". This can significantly slow sampling.
- Do not switch container-level public transforms to shallow copy without a clear immutability contract.

## Maintenance Notes

- If new waveform/container fields are added, re-check whether shallow-copy sites still touch only scalar modifiers.
- If `PulseArray` or `PulseSchedule` move to immutable internals in the future, revisit deep-copy requirements for their public transforms.
