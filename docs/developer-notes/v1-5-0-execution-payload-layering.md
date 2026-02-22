# v1.5.0 Execution Payload Layering

## Status

- State: `IMPLEMENTED`
- Documented on: 2026-02-19

## Problem

`Quel1ExecutionPayload` and `Quel3ExecutionPayload` were defined in different layers:

- `Quel1ExecutionPayload`: backend layer
- `Quel3ExecutionPayload`: measurement adapter layer

This created an inverted dependency in the QuEL-3 path where backend code imported
measurement-layer payload types.

## Root cause

- QuEL-1 path was migrated from legacy qubecalib integration and kept a backend-local payload.
- QuEL-3 support was added later through measurement adapter scaffolding and introduced payload
  dataclasses in adapter code.
- `BackendExecutionRequest.payload` is intentionally generic (`Any`), so payload ownership was
  not enforced by type boundaries.

## Target architecture

Execution payloads are backend contracts and should be defined in backend modules.

- Direction rule: `measurement -> backend`
- No reverse dependency from backend to measurement for payload models
- Measurement adapters build backend payloads
- Backend executors/controllers consume backend payloads

## Implemented changes

### 1) Move QuEL-3 payload models to backend layer

- Added `src/qubex/backend/quel3/quel3_execution_payload.py`:
  - `Quel3CaptureWindow`
  - `Quel3WaveformEvent`
  - `Quel3TargetTimeline`
  - `Quel3ExecutionPayload`
- Exported these in `src/qubex/backend/quel3/__init__.py`.

### 2) Remove backend -> measurement dependency in QuEL-3 path

- Updated `src/qubex/backend/quel3/quel3_backend_controller.py` to use backend-local payload types.
- Updated `src/qubex/backend/quel3/quel3_sequencer_compiler.py` to use backend-local payload types.
- Updated `src/qubex/backend/quel3/quel3_backend_executor.py` as backend-local executor for
  QuEL-3 payload execution.
- This removes direct payload imports from `qubex.measurement.adapters.backend_adapter`.

### 3) Keep measurement layer focused on schedule-to-payload conversion

- `Quel3MeasurementBackendAdapter` stays in `qubex.measurement.adapters`.
- `Quel3BackendExecutor` is owned by `qubex.backend.quel3` and imported by
  `MeasurementScheduleRunner` from backend layer.

### 4) Align QuEL-1 payload shape with backend-plan pattern

`Quel1ExecutionPayload` now carries sequencer compilation inputs instead of a pre-built sequencer:

- `gen_sampled_sequence`
- `cap_sampled_sequence`
- `resource_map`
- `interval`
- execution options (`repeats`, `integral_mode`, DSP/classifier settings)

`Quel1BackendExecutor` now compiles a sequencer via
`backend_controller.create_quel1_sequencer(...)` right before execution.

This keeps adapter responsibility at "build execution plan" and backend responsibility at
"compile and execute".

## Compatibility notes

- Measurement-layer exports currently keep backward-compatible re-export of QuEL-3 payload classes
  through `qubex.measurement.adapters` and `qubex.measurement`.
- Canonical ownership is now backend (`qubex.backend.quel3`).

## Validation

- `uv run ruff check`
- `uv run pyright`
- `uv run pytest`

All checks pass after the payload-layering refactor.
