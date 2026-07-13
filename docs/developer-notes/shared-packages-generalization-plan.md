# Shared Packages Generalization Plan

## Status

- State: `ACTIVE`
- Last updated: 2026-02-20

## Objective

- Keep `qxvisualizer` and `qxfitting` generic, reusable, and decoupled from experiment-domain terminology and backend-specific behavior.
- Let domain packages (`qubex`, `qxpulse`, `qxsimulator`) own domain naming and orchestration while using shared-package primitives.

## Scope

- In scope:
  - API and module boundaries of `qxvisualizer` and `qxfitting`
  - progressive extraction of generic helpers from domain packages
  - compatibility strategy for existing `qubex` import paths
- Out of scope:
  - one-shot breaking API cleanup across all call sites
  - immediate introduction of `qxanalysis`

## Guardrails

### Naming and semantics

- Do not introduce domain terms in shared-package public APIs:
  - hardware/backend names (`quel`, `backend`, controller-specific words)
  - experiment-target terms (`qubit`, `resonator`, `capture`, channel labels like `Q00`, `RQ00`)
- Prefer neutral terms:
  - `series`, `samples`, `labels`, `matrix`, `vectors`, `fit_result`

### Responsibility split

- `qxvisualizer`:
  - Plotly-only builders and plotting wrappers
  - style/template/config and shared figure helpers
- `qxfitting`:
  - starts as a placeholder package for new fitting APIs
  - new APIs are added incrementally with domain-neutral naming
- Domain packages:
  - translate domain objects to generic arrays/inputs
  - choose domain wording for titles/annotations

### Compatibility

- Keep `qubex.analysis.fitting` as the legacy compatibility baseline during migration.
- Do not add dynamic re-export/delegation between `qubex` and `qxfitting`.
- Shared package refactors must not force domain API changes in the same step.

## Work Plan

### Phase 1: Baseline and compatibility lock (done / ongoing)

- [x] Keep full legacy fitting implementation in `qubex.analysis.fitting`.
- [x] Keep `qxfitting` importable as placeholder.
- [x] Add compatibility tests for the baseline and placeholder state.

### Phase 2: `qxvisualizer` generalization

- [ ] Inventory shared plotting utilities and classify:
  - generic (belongs in `qxvisualizer`)
  - domain-specific (stays in domain package)
- [ ] Move generic constants from domain packages to `qxvisualizer`:
  - only names that are not tied to pulse/backend semantics
- [ ] Ensure public functions follow consistent pair:
  - `make_*_figure(...) -> go.Figure`
  - `plot_*(...) -> None`
- [ ] Keep external visualization backends outside `qxvisualizer`.

### Phase 3: `qxfitting` generalization

- [ ] Add new fit APIs in `qxfitting` without changing legacy `qubex` APIs.
- [ ] Split internals into:
  - `core` (pure fitting logic)
  - `plotting` (figure builders using `qxvisualizer`)
- [ ] Replace domain-specific parameter names in public APIs where feasible:
  - e.g., `target` -> `label` (with backward-compatible aliases)
- [ ] Introduce explicit initial-guess APIs:
  - `guess_*` helpers
  - multi-start options in fit entry points
- [ ] Standardize fit diagnostics payload:
  - solver status, selected initial guess, retry counts, residual summary

### Phase 4: Domain call-site cleanup

- [ ] Update domain packages to call shared generic APIs first, then add domain labeling.
- [ ] Remove duplicated style/constants/helpers from domain layers once replaced.
- [ ] Keep compatibility wrappers until migration is complete and announced.

## Concrete Checks

Run these checks during each migration PR.

### Static quality

- `uv run ruff check`
- `uv run pyright`
- `uv run pytest`

### Shared-package vocabulary audit

- Review public names in:
  - `packages/qxvisualizer/src/qxvisualizer`
  - `packages/qxfitting/src/qxfitting`
- Reject new public symbols that encode domain/backend terms.

### Compatibility

- Verify `qubex.analysis.fitting` baseline tests pass.
- Verify `qxfitting` placeholder/new API contract tests pass.
- Verify representative domain services still call fitting APIs unchanged.

## Exit Criteria

- `qxvisualizer` public API is domain-neutral and Plotly-focused.
- `qxfitting` public API is fit-centric and domain-neutral.
- Domain naming remains in domain packages only.
- `qubex.analysis.fitting` remains functional as legacy compatibility baseline until planned deprecation.
