# v1.5.0 Experiment async measurement API plan

## Status

- State: `IN_PROGRESS`
- Created: `2026-02-25`
- Updated: `2026-03-02`

## Implementation snapshot (2026-03-02)

- Implemented in source:
  - `Experiment.run_measurement(...)`
  - `Experiment.run_sweep_measurement(...)`
- Delegation tests are present for async and legacy compatibility surfaces.
- Remaining work before beta documentation freeze:
  - user-guide async-first + legacy compatibility section update
  - explicit migration notes for v1.5.0 users

## Goal

Add async-first measurement entrypoints to `Experiment` in v1.5.0:

- `async def run_measurement(...)`
- `async def run_sweep_measurement(...)`

Treat these as the primary direction for new usage, while preserving existing synchronous APIs for compatibility.

## Non-goals

- Finalize full signature and return schema in this document.
- Remove existing synchronous APIs in v1.5.x.

## Compatibility policy (planned)

- Keep existing APIs available in v1.5.x as legacy compatibility surface:
  - `measure`
  - `execute`
  - `sweep_parameter`
  - `sweep_measurement` (including service-level legacy path)
- Legacy APIs remain documented for users.
- Legacy behavior and return types should remain stable while internals migrate.

## Candidate API shape (draft)

```python
class Experiment:
    async def run_measurement(...): ...
    async def run_sweep_measurement(...): ...
```

Notes:

- Signatures are implemented in source and treated as beta baseline.
- Delegation should follow existing ownership:
  - `Experiment` facade -> `MeasurementService` -> measurement-layer execution services.

## Migration direction

1. Define minimal signature contracts and error semantics.
2. Implement async methods on `Experiment` as thin delegation APIs.
3. Route legacy synchronous APIs through the new execution path where behavior matches.
4. Keep user documentation split into:
   - recommended async APIs
   - legacy synchronous APIs (compatibility section).

## Implementation checklist

- [x] Freeze `run_measurement` / `run_sweep_measurement` signatures.
- [x] Add `Experiment` delegation tests for the new async APIs.
- [x] Add compatibility tests that keep legacy API behavior unchanged.
- [ ] Update user-guide pages to show async-first examples and legacy compatibility notes.
- [ ] Add v1.5.0 migration notes for users.

## Open questions (remaining)

- Should `Experiment` provide synchronous helper wrappers around coroutine APIs for notebook ergonomics?
- How should cancellation/timeout/error handling be surfaced relative to current sync APIs?
