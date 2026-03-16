# v1.5.0 PulseSchedule Per-Channel Sampling-Period Design Draft

## Status

- State: `PROPOSED`
- Drafted on: `2026-02-20`
- Owner: pulse/measurement maintainers
- v1.5.0 scope decision: `OUT_OF_SCOPE` (documentation only, no implementation in v1.5.0)

## Purpose

Define requirements, implementation policy, and decision items for evolving
`qxpulse.PulseSchedule` from a single sampling-period assumption to
per-channel sampling-period handling.

## Background

Current `PulseSchedule` behavior is effectively single-`dt`:

- `PulseSchedule.length` derives sample length from global `Waveform.SAMPLING_PERIOD`.
- `PulseSchedule.is_valid()` checks equality of per-channel sample lengths.
- `barrier()` pads channels via `Blank(duration=diff)` without explicit channel `dt`.
- Multiple code paths assume `schedule.length` is globally meaningful.

With mixed channel sampling periods, these assumptions break.

## Scope

In scope:

- Per-channel `sampling_period` ownership in `PulseSchedule`.
- Constructor-time explicit channel sampling-period declaration.
- First-waveform inference when channel `dt` is not declared.
- Same-channel mismatch detection and failure.
- Barrier blank insertion that is correct with mixed channel `dt`.

Out of scope for this draft:

- Backend-specific waveform resampling.
- Mixed-`dt` execution support policy for every backend (covered by decision items).

## v1.5.0 Scope Decision (2026-02-20)

- Per-channel `sampling_period` implementation in `PulseSchedule` is deferred to a post-v1.5.0 release.
- v1.5.0 deliverable is documentation only:
  - requirements
  - implementation policy options
  - unresolved decision items
- No API/behavior change is planned for v1.5.0 in:
  - `PulseSchedule` constructor/add/barrier semantics
  - `PulseSchedule.length`/`is_valid` semantics
  - mixed-`dt` backend acceptance behavior

## Temporary QuEL-3 Workaround (2026-03-16)

While `PulseSchedule` still behaves as single-`dt`, the QuEL-3 path keeps
control waveforms on `0.4 ns` and normalizes readout waveforms to `0.8 ns`
inside the QuEL-3 measurement adapter before quelware registration.

This is a temporary backend-local workaround, not the intended long-term
sampling-period model.

## Requirements

### R-001 Channel sampling-period registry

`PulseSchedule` must maintain `sampling_period` per channel (resolved or unresolved).

### R-002 Explicit constructor declaration

At construction time, users must be able to declare channel `sampling_period`
explicitly (API shape is a decision item).

### R-003 First-add inference

If a channel `sampling_period` is not declared, the first added waveform on that
channel defines it.

### R-004 Same-channel consistency

After a channel `sampling_period` is resolved, adding a waveform with a
different `sampling_period` to the same channel must raise `ValueError`.

This applies to:

- direct `Pulse`/`Waveform` adds
- `PulseArray` adds (all contained waveforms must be consistent with channel `dt`)

### R-005 Barrier with mixed sampling periods

`barrier()` must preserve wall-clock synchronization in ns across target channels,
while generating blank waveforms compatible with each channel `dt`.

### R-006 Backward compatibility

Existing single-`dt` schedules must keep current behavior and not require code
changes by users.

### R-007 Deterministic errors

When alignment or inference is impossible, error messages must include:

- channel label
- expected/resolved `dt`
- incoming `dt` or requested duration

## Implementation Policy

### P-001 Internal data model

- Add `channel_sampling_periods: dict[str, float | None]` in `PulseSchedule`.
- Treat `None` as unresolved.
- Keep offsets in ns (`_offsets`) as source of truth for schedule timing.

### P-002 `add()` resolution and validation

On `add(label, obj)`:

- ensure channel exists
- inspect waveform `dt` values in `obj`
- if channel `dt` unresolved:
  - resolve from waveform `dt`
- else:
  - validate incoming waveform `dt == channel_dt` within tolerance
- if mismatch:
  - raise `ValueError`

For `PhaseShift`-only add:

- do not resolve channel `dt`
- append normally

### P-003 Barrier timing policy

Barrier alignment is defined in wall-clock time:

- Let `t0 = max(offset[label] for label in barrier_labels)`.
- Compute `t_barrier` as the minimum `t >= t0` that is representable on all
  participating resolved channel grids.
- For each channel, add `Blank(duration=t_barrier - offset[label], sampling_period=channel_dt)`.

If `t0` is already representable for all channels, `t_barrier = t0`.

### P-004 Handling unresolved channels in barrier paths

For channels with unresolved `dt`, barrier padding cannot directly create
`Blank(...)` safely. Policy requires explicit behavior (decision item).

### P-005 API compatibility for `length` and validity

Mixed-`dt` schedules cannot rely on a single global sample length.
Policy needs explicit semantics for:

- `PulseSchedule.length`
- `PulseSchedule.is_valid()`
- `get_sampled_sequences(duration=...)` when requested duration is not on every
  channel grid

### P-006 Backend safety policy

Backend adapters/builders that require uniform sample grid must gate mixed-`dt`
schedules explicitly and fail with clear diagnostics.

## Required Decisions

### D-001 Constructor API for explicit channel `dt`

Options:

1. Add `channel_sampling_periods: Mapping[str, float] | None` to `PulseSchedule.__init__`.
2. Extend `PulseChannel` with `sampling_period: float | None` and rely on
   `channels=[PulseChannel(...)]`.
3. Support both 1 and 2.

Recommendation: `3` (keep ergonomic map + structured channel metadata path).

### D-002 Unresolved channel participates in `barrier()`

Options:

1. Error immediately if padding is needed before channel `dt` is resolved.
2. Track pending leading blank in ns, materialize when first waveform resolves `dt`.
3. Force fallback to global default `dt` (legacy behavior).

Recommendation: `2` for flexibility without hidden default coupling.

### D-003 Barrier common-grid alignment algorithm

Options:

1. Require exact representability at `t0`; error if not representable.
2. Round up to nearest common representable `t_barrier` (ceil policy).
3. Resample one or more channels.

Recommendation: `2` (monotonic and practical; no resampling in scheduler core).

### D-004 Semantics of `PulseSchedule.length`

Options:

1. Keep as-is and raise when mixed `dt`.
2. Redefine as max channel sample length.
3. Deprecate and replace with explicit per-channel length APIs.

Recommendation: `1` now, `3` as planned cleanup in a later version.

### D-005 Semantics of `PulseSchedule.is_valid()`

Options:

1. Keep sample-length equality check (fails mixed-`dt` by design).
2. Change to duration-equality check in ns.
3. Split into `is_time_aligned()` and `is_uniform_grid()`.

Recommendation: `3` to avoid ambiguous meaning and keep strict checks available.

### D-006 `plot(time_unit=\"samples\")` with mixed `dt`

Options:

1. Disallow and require `time_unit=\"ns\"`.
2. Keep and show per-channel sample axis independently.
3. Keep legacy behavior and document undefined semantics.

Recommendation: `1` (clear and predictable).

### D-007 Backend acceptance policy for mixed-`dt` schedules

Options:

1. Reject mixed-`dt` for QuEL-1 adapter/builder paths.
2. Auto-resample to backend `dt`.
3. Allow only in QuEL-3 timeline/event path, reject in QuEL-1 strict path.

Recommendation: `3` (aligned with current strict vs relaxed backend split).

### D-008 Common-grid calculation robustness (`dt` LCM growth / precision)

Problem:

- When per-channel `dt` values differ, barrier alignment may require a common grid.
- Naive `float` arithmetic can mis-detect representability.
- LCM of heterogeneous periods can grow too large and create excessive blank insertion.

Options:

1. Keep `float`-only checks with tolerance-based representability.
2. Normalize `dt` to integer ticks on a fixed quantum and compute LCM on integers.
3. Avoid global common-grid and use backend-specific resampling in schedule layer.

Recommendation: `2` with explicit LCM-cap guardrails.

v1.5.0 decision:

- Document only, do not implement.
- Track as post-v1.5.0 backlog item.

## Proposed Rollout (post-v1.5.0)

1. Phase A: data model + validation (`add`, constructor, channel `dt` accessors).
2. Phase B: barrier/pad logic with mixed-`dt` alignment and tests.
3. Phase C: API semantics (`length`, validity predicates, plotting guardrails).
4. Phase D: measurement/backend integration guardrails and docs update.

## Test Plan Outline

- Unit: `tests/pulse/test_pulse_schedule.py` add mixed-`dt` cases.
- Unit: barrier fill with unequal `dt`, including ceil-to-common-grid behavior.
- Unit: mismatch errors for same-channel conflicting `dt`.
- Integration: measurement path rejects or accepts mixed-`dt` by backend policy.

## Acceptance Criteria

- Requirements `R-001` to `R-007` are covered by tests.
- Single-`dt` existing tests remain green without behavior regression.
- Decision items `D-001` to `D-007` are resolved and reflected in docs/API.
