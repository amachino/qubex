# v1.5.0 Shared Pulse Factory Design

## Status

- State: `PROPOSED`
- Documented on: 2026-02-18
- Earliest implementation start: 2026-02-19

## Background

- `qxpulse` manages sampling period as module-level global state.
- Sampling period is determined by selected backend/controller.
- Both `Experiment` and `MeasurementClient` must follow the same backend-defined sampling period.
- Current mitigation (`ExperimentContext` syncing `set_sampling_period`) reduces mismatch risk but still relies on mutable global state.

## Problem Statement

- Multiple `Experiment` instances in the same Python process can overwrite each other's pulse sampling period.
- Direct pulse construction (`FlatTop(...)`, `Drag(...)`, etc.) is scattered across services and contrib code, making ownership of sampling period implicit.
- Backend-defined sampling period can drift from pulse-construction context unless every call site is carefully controlled.

## Goals

- Make backend/session context the owner of pulse sampling semantics.
- Centralize pulse construction behind a shared factory used by both `Experiment` and `MeasurementClient`.
- Keep QuEL-1 behavior unchanged while supporting non-2ns backends.
- Enable incremental migration without large one-shot rewrites.

## Non-goals

- Modifying `qxpulse` internals in v1.5.0.
- Removing all global-state interaction in one phase.
- Redesigning measurement-layer factories (`MeasurementPulseFactory`) in this step.

## Proposed Architecture

### 1) New component: `SharedPulseFactory`

- File candidate: `src/qubex/pulse/shared_pulse_factory.py`
- Responsibility: construct qxpulse objects under backend/session-owned sampling period.
- Input:
  - Sampling period provider (`float` or callable).
- Output:
  - qxpulse waveforms/pulses (`Blank`, `FlatTop`, `Drag`, `Rect`, `CrossResonance`, etc.).

### 2) Sampling-period guard scope

- Factory methods run inside a guarded scope:
  1. Save current global sampling period.
  2. Set experiment sampling period.
  3. Construct pulse/waveform.
  4. Restore previous sampling period.
- Optional: add lock around scope if multi-thread execution is in scope for notebook workflows.

### 3) Ownership and integration points

- Owner: backend/session scope (`SystemManager` or backend manager layer), not `Experiment` layer.
- `ExperimentContext` and `MeasurementClient` both obtain the same shared factory instance.
- `PulseService` and measurement-side pulse builders call shared factory methods instead of direct qxpulse constructors.
- Existing `ExperimentContext._sync_pulse_sampling_period()` remains as compatibility bridge during migration.

### 4) Sampling period source policy

- Canonical source: backend/controller contract (`DEFAULT_SAMPLING_PERIOD`).
- Fallback order:
  1. `measurement.sampling_period` (resolved from backend contract)
  2. `backend_controller.DEFAULT_SAMPLING_PERIOD`
  3. QuEL-1 default (`2.0 ns`)

## Migration Plan

### Phase A (first implementation pass)

- Introduce `SharedPulseFactory` and session-level wiring.
- Switch `PulseService` core methods first:
  - `get_hpi_pulse`
  - `get_pi_pulse`
  - `get_drag_hpi_pulse`
  - `get_drag_pi_pulse`

### Phase B

- Replace direct qxpulse constructors in:
  - `measurement_pulse_factory.py`
  - `calibration_service.py`
  - `characterization_service.py`
  - `optimization_service.py`
  - selected `contrib/` modules

### Phase C

- Add regression guard:
  - CI grep/lint rule for direct `FlatTop(`/`Drag(`/`Rect(` usage inside experiment-layer services (except factory module).

## Test Strategy

- Unit: factory scope restores previous global sampling period.
- Unit: factory uses measurement/backend-derived sampling period.
- Integration: two experiment instances with different sampling periods do not leak through factory calls.
- Integration: `Experiment` and `MeasurementClient` share the same backend-derived factory behavior.
- Regression: existing pulse-service delegation and calibration behavior stay unchanged in QuEL-1 (`2.0 ns`).

## Risks and Mitigations

- Risk: hidden direct constructors remain.
  - Mitigation: phase-C grep guard and incremental module audit.
- Risk: temporary global switch overhead.
  - Mitigation: narrow scope and avoid per-sample calls.
- Risk: thread contention if notebooks run concurrent experiments.
  - Mitigation: optional lock in factory scope (decision pending).

## Open Questions

- Do we require thread-safety lock in v1.5.0 beta, or can it be deferred to GA?
- Should `MeasurementPulseFactory` be folded into `SharedPulseFactory` or remain a thin adapter on top?
- At what phase do we deprecate `ExperimentContext._sync_pulse_sampling_period()`?
