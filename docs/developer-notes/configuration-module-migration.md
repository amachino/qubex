# Configuration Module Migration Plan

## Goal

Reduce `qubex.backend` responsibility and move configuration concerns into
`qubex.configuration` while keeping runtime behavior stable for v1.5.0 work.

## Problem statement

- `ConfigLoader` currently mixes:
  - file IO,
  - schema normalization (`wiring.yaml` and `wiring.v2.yaml`),
  - runtime model assembly (`QuantumSystem`, `ControlSystem`, `ExperimentSystem`).
- `SystemManager` also contains wiring-file selection logic.
- As QuEL-3 configuration grows, keeping all of this under `backend` increases coupling.

## Target package split

- `qubex.configuration`
  - YAML schema handling and normalization
  - loader orchestration APIs
- `qubex.backend`
  - hardware/runtime control (`SystemManager`, backend controllers, executors)
- domain model layer (existing `backend` models for now)
  - `Chip`, `ControlSystem`, `WiringInfo`, `ExperimentSystem`

## Migration principles

- Keep behavior-compatible increments.
- Keep one migration step per commit-sized change.
- Keep `Measurement` compatibility surface unchanged.
- Prefer extraction over rewrite.

## Phase plan

### Phase 1 (completed)

- Added `qubex.configuration` package.
- Moved wiring normalization utilities to:
  - `src/qubex/configuration/wiring.py`
- `ConfigLoader` now delegates wiring-v2 normalization to configuration module.
- Added focused tests for configuration wiring helpers.

### Phase 2 (in progress)

- Moved `ConfigLoader` implementation to `qubex.configuration.config_loader`.
- Kept `qubex.backend.config_loader` as compatibility shim during v1.5.0 pre-release.
- Updated internal imports (`SystemManager`, diagnostics) to configuration namespace.
- Added import-compatibility tests for `ConfigLoader` exports.

### Phase 3 (in progress)

- Introduced explicit `ConfigLoader.load()` lifecycle API.
- Added `autoload` option (`True` by default) for transition compatibility.
- Updated `SystemManager` to use explicit `ConfigLoader(..., autoload=False)` then `load()`.
- Wiring-file selection policy is still in `SystemManager` for v1.5.0 and will move later.

## Hardware regression timing (for this migration)

Run minimal real-hardware checks at the following points:

1. After Phase 2 namespace move
   - Verify load/connect/sync still works with existing QuEL-1 flow.
   - Verify `backend_kind="quel3"` load path selects expected wiring file.
2. After Phase 3 load-lifecycle change
   - Verify one full session bootstrap path (`load -> connect -> execute minimal measurement`) on target environment.
   - Confirm no regression in `mock_mode=True` setup path.
3. Before release cut
   - Run standard hardware validation template for both backend families.
   - Record environment, `dt`, and pass/fail in validation notes.

## Current status

- Runtime behavior remains unchanged for existing tests.
- `wiring.v2.yaml` support remains active.
- `SystemManager.load(..., backend_kind="quel3")` still prefers `wiring.v2.yaml` when present.
