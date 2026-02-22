# Measurement backend responsibility and execution-path policy

## Status

- State: `PROPOSED`
- Documented on: 2026-02-22

## Purpose

Define a unified responsibility model for QuEL-1 and QuEL-3 measurement execution paths before v1.5.0 release freeze.

## Scope

- In scope:
  - Measurement execution path from `MeasurementClient` to backend execution result.
  - Responsibility boundaries between measurement layer and backend layer.
  - Naming and placement policy for classes in QuEL-1 and QuEL-3 paths.
- Out of scope:
  - Behavioral changes in user-facing measurement APIs.
  - Hardware protocol changes in qubecalib or quelware-client.

## Terms

- `Control Plane`:
  - The layer that handles device/session control operations such as load, connect, synchronize, configure, and state pull/push.
  - It should not own measurement result decoding policy.
- `Measurement Pipeline`:
  - The layer that handles one measurement execution flow:
    1. validate schedule,
    2. compile backend request,
    3. run request,
    4. decode canonical `MeasurementResult`.
- `API`:
  - Entry points for application users (for example `MeasurementClient`).
- `SPI` (`Service Provider Interface`):
  - Internal extension contracts implemented by backend-specific providers (for example `MeasurementPipeline` and `BackendControlPlane` interfaces).

## Current issues

- QuEL-3 measurement flow is partially hosted in controller implementation, while QuEL-1 flow uses adapter + executor + result factory split.
- QuEL-3 controller currently inherits from QuEL-1 controller, which couples control-plane concerns and backend-specific execution concerns.
- Result conversion path is asymmetric:
  - QuEL-1 uses factory conversion from raw backend result.
  - QuEL-3 can return canonical result directly from controller.

## Design principles

- Keep one common execution model for QuEL-1 and QuEL-3.
- Isolate backend-specific behavior behind SPI contracts.
- Separate control-plane concerns from execution/data-path concerns.
- Keep `MeasurementClient` API stable and backend-agnostic.
- Keep ownership direction explicit:
  - `measurement -> backend SPI contracts`
  - `backend implementations -> backend-specific modules`

## Target architecture

### Layer model

1. API layer:
   - `MeasurementClient` (public entry point).
2. Use-case orchestration layer:
   - `MeasurementExecutionService` (backend-agnostic execution coordinator).
3. SPI contracts:
   - `MeasurementPipeline` (validate/compile/run/decode).
   - `BackendControlPlane` (load/connect/sync/configure lifecycle).
4. Backend implementations:
   - `quel1` implementation set.
   - `quel3` implementation set.

### Unified execution sequence

1. Build schedule and config in measurement layer.
2. Resolve active backend session from `SystemManager`.
3. Execute through selected `MeasurementPipeline`.
4. Return canonical `MeasurementResult`.
5. Convert to legacy result model only at API compatibility boundary when required.

## Responsibility boundaries

### `BackendControlPlane` responsibilities

- Backend family selection and controller/session lifecycle.
- Hardware configuration sync (`load`, `connect`, `pull`, `push`).
- Box/port/channel/target control-plane operations.

### `MeasurementPipeline` responsibilities

- Schedule validation according to backend constraint profile.
- Backend request compilation from canonical schedule/config.
- Backend execution invocation for prepared request.
- Canonical result decoding.

### Explicit non-responsibilities

- `ControlPlane` must not perform measurement result decoding logic.
- `MeasurementPipeline` must not own full-system control synchronization logic.

## Class organization policy

### Common contracts and orchestration

- `src/qubex/backend/spi/control_plane.py`
  - `BackendControlPlane` protocol.
- `src/qubex/backend/spi/measurement_pipeline.py`
  - `MeasurementPipeline` protocol.
- `src/qubex/backend/session.py`
  - `BackendSession` object carrying:
    - `backend_kind`,
    - `control_plane`,
    - `measurement_pipeline`,
    - `constraint_profile`.
- `src/qubex/measurement/execution_service.py`
  - `MeasurementExecutionService`.

### QuEL-1 implementation policy

- `src/qubex/backend/quel1/control_plane.py`
  - QuEL-1 control-plane implementation.
- `src/qubex/backend/quel1/measurement/pipeline.py`
  - QuEL-1 measurement pipeline.
- `src/qubex/backend/quel1/measurement/compiler.py`
  - Request compilation for QuEL-1.
- `src/qubex/backend/quel1/measurement/runner.py`
  - Request runner for QuEL-1.
- `src/qubex/backend/quel1/measurement/decoder.py`
  - Raw-result to canonical-result decoder for QuEL-1.

### QuEL-3 implementation policy

- `src/qubex/backend/quel3/control_plane.py`
  - QuEL-3 control-plane implementation.
- `src/qubex/backend/quel3/measurement/pipeline.py`
  - QuEL-3 measurement pipeline.
- `src/qubex/backend/quel3/measurement/compiler.py`
  - Request compilation for QuEL-3 fixed timeline.
- `src/qubex/backend/quel3/measurement/runner.py`
  - Request runner for QuEL-3 quelware execution.
- `src/qubex/backend/quel3/measurement/decoder.py`
  - QuEL-3 result decoding to canonical result.

## Naming policy

- Prefer `*ControlPlane` for lifecycle/config synchronization implementations.
- Prefer `*Pipeline` for measurement execution orchestrators.
- Prefer `*Compiler`, `*Runner`, `*Decoder` for single-purpose execution components.
- Keep backend-family prefixes explicit (`Quel1*`, `Quel3*`).
- Avoid using one backend family class as a parent class for another backend family.

## Migration plan

### Phase 0: Introduce contracts and wrappers (no behavior change)

- Add SPI protocols and `MeasurementExecutionService`.
- Keep existing classes as wrappers/adapters to the new contracts.

### Phase 1: Split QuEL-3 execution concerns from current controller

- Extract QuEL-3 measurement execution and result decoding into pipeline components.
- Keep QuEL-3 control-plane class focused on lifecycle/config operations.

### Phase 2: Align QuEL-1 and QuEL-3 structure

- Align both families to compiler + runner + decoder structure.
- Remove special-case execution branches from orchestration layer.

### Phase 3: Remove inheritance coupling

- Replace `Quel3BackendController(Quel1BackendController)` inheritance with composition or independent implementation.

### Phase 4: Deprecate old names

- Keep backward-compatible aliases during v1.5.x.
- Remove deprecated aliases in the next planned major/minor cleanup window.

## Acceptance criteria

- A single common execution sequence exists for QuEL-1 and QuEL-3.
- Measurement pipeline contracts are explicit and test-covered.
- Control-plane and execution responsibilities are separated with no cross-layer inversion.
- `MeasurementClient` API remains compatible at v1.5.0 contract scope.
