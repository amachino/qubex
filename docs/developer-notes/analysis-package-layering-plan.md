# Analysis package layering plan

## Status

- State: `ACTIVE`
- Last updated: 2026-02-20

## Decision summary

- Keep `qubex.analysis.fitting` as the legacy implementation and freeze it as the compatibility baseline.
- Keep `qxfitting` as a placeholder package, then add new fitting APIs there incrementally.
- Migrate domain call sites from `qubex.analysis.fitting` to `qxfitting` step by step.
- Remove `qubex.analysis.fitting` only after migration is complete and verified.

## Why this policy

- Dynamic re-export from `qubex` to `qxfitting` couples release timing and increases accidental break risk.
- The legacy `qubex` implementation is already widely used in domain services and notebooks.
- A staged migration keeps user-facing behavior stable while allowing `qxfitting` API redesign.

## Responsibility boundaries

### `qubex.analysis.fitting` (legacy baseline)

- Owns current fitting behavior and output semantics.
- Accepts bug fixes only.
- Does not accept new fitting APIs.

### `qxfitting` (new API track)

- Starts from a minimal placeholder surface.
- Adds new fitting APIs with explicit, domain-neutral names.
- Must not depend on `qubex`.

### Domain packages (`qubex` services/contrib)

- Continue using `qubex.analysis.fitting` until each flow has a tested `qxfitting` replacement.
- Migrate call sites one feature area at a time.

## Migration stages

### Stage 1: Freeze baseline (done)

- Keep full legacy implementation in `qubex.analysis.fitting`.
- Keep `qxfitting` importable but minimal.
- Keep compatibility tests green.

### Stage 2: Introduce `qxfitting` v-next APIs

- Add new fit functions in `qxfitting` without changing legacy names by default.
- Define stable return contracts for new APIs.
- Add focused unit tests under `packages/qxfitting` as APIs are introduced.

### Stage 3: Domain-by-domain migration

- Migrate one domain workflow at a time (for example: RB, Rabi, Ramsey).
- Keep each migration PR small and include behavior checks.
- Do not mix broad refactors with API migrations in one PR.

### Stage 4: Deprecation and removal

- After all production call sites migrate, announce deprecation window.
- Remove `qubex.analysis.fitting` only after the deprecation window and contract checks pass.

## Guardrails

- No new dynamic delegation (`__getattr__`) between `qubex` and `qxfitting`.
- New fitting development must target `qxfitting`, not `qubex.analysis.fitting`.
- Compatibility-sensitive changes must include:
  - affected API list
  - migration impact
  - test updates

## Exit criteria

- Domain call sites no longer depend on `qubex.analysis.fitting`.
- `qxfitting` provides the required stable API set.
- Legacy module removal is completed with release notes and migration guidance.
