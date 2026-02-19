# Analysis Package Layering Plan

## Status

- State: `PROPOSED`
- Documented on: 2026-02-19

## Decision Summary

- First split fitting logic into a dedicated package: `qxfitting`.
- Keep `qubex.analysis.fitting` as a compatibility wrapper during migration.
- If additional analysis domains grow later, introduce `qxanalysis` as a higher-level package.

## Background

- `qubex.analysis` currently mixes pure analysis logic, plotting, and notebook/UI-oriented paths.
- Visualization responsibilities are being centralized into `qxvisualizer`.
- Fitting workflows need internal refactoring (especially initial-value estimation strategy), and package boundaries should support that work.

## Naming Decision

- Adopt `qxfitting` now.
- Do not introduce `qxanalysis` yet because the scope is currently too broad for one package.
- Keep `qxanalysis` as a future umbrella option once multiple stable analysis domains exist.

## Responsibility Boundaries

### `qxfitting` (new)

- Fitting models and solvers.
- Initial-value estimation and multi-start strategies.
- Fit result containers and diagnostics.
- Plot helpers for fitting results, implemented using `qxvisualizer`.

### `qubex.analysis.fitting` (compat layer)

- Re-export `qxfitting` public APIs.
- Preserve existing import paths for users during transition.
- Carry deprecation notices only when migration timing is finalized.

### `qxanalysis` (future)

- Optional higher-level package that can aggregate multiple analysis domains:
  - fitting
  - tomography
  - spectral/statistical analysis
  - other stable analysis modules

## Dependency Direction

- `qxvisualizer` is the visualization base package.
- `qxfitting` may depend on `qxvisualizer`.
- `qubex` depends on `qxfitting` and `qxvisualizer`.
- Avoid reverse dependencies (`qxvisualizer` must not depend on `qxfitting`).

## Migration Plan

### Phase 1

- Create `qxfitting` package skeleton.
- Move fitting core and plotting modules from `qubex.analysis.fitting`.
- Keep behavior compatible with current public APIs.

### Phase 2

- Convert `qubex.analysis.fitting` into compatibility re-exports.
- Add focused migration tests to ensure old import paths still work.

### Phase 3

- Refactor fitting internals (initial-value estimation, multi-start, diagnostics) inside `qxfitting`.
- Keep output/plot semantics compatible unless explicitly versioned.

### Phase 4 (future)

- Re-evaluate whether `qxanalysis` should be introduced as an umbrella package.

## Non-goals

- Creating `qxanalysis` immediately.
- Performing a one-shot breaking rename of user-facing import paths.
- Migrating all non-fitting analysis modules in the same step.
