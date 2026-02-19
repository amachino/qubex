# Shared Packages Generalization Plan

## Status

- State: `PROPOSED`
- Documented on: 2026-02-19

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
  - fit models, initial-guess logic, solver orchestration, diagnostics
  - fit-oriented plot builders/wrappers implemented via `qxvisualizer`
- Domain packages:
  - translate domain objects to generic arrays/inputs
  - choose domain wording for titles/annotations

### Compatibility

- Keep `qubex.analysis.fitting` as compatibility wrapper during migration.
- Shared package refactors must not force domain API changes in the same step.

## Work Plan

### Phase 1: Baseline and compatibility lock (done / ongoing)

- [x] Create `qxfitting` package and migrate current fitting implementation.
- [x] Keep `qubex.analysis.fitting` as wrapper to `qxfitting`.
- [x] Add wrapper compatibility tests.

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

- Verify `qubex.analysis.fitting` wrapper tests pass.
- Verify representative domain services still call fitting APIs unchanged.

## Exit Criteria

- `qxvisualizer` public API is domain-neutral and Plotly-focused.
- `qxfitting` public API is fit-centric and domain-neutral.
- Domain naming remains in domain packages only.
- `qubex.analysis.fitting` remains functional as compatibility layer until planned deprecation.
