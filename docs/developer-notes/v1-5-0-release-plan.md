# v1.5.0 Release Plan

## Release target

- Beta release window: by February 2026
- GA release window: by March 2026

## Dependency status

- `quelware-client` is not yet complete for final QuEL-3 execution-path decisions.
- QuEL-3-specific implementation lock items remain `PROPOSED` and are resumed after dependency completion.

## Scope

- Add support for QuEL-3 controller using new `quelware-client`
- Keep backward compatibility with existing controllers
- Primary compatibility target is `MeasurementClient` level
- `Experiment` compatibility is expected to be preserved through `MeasurementClient` compatibility
- Below `MeasurementClient`, implementation may diverge for QuEL-3 specific behavior
- Remove assumptions tied to fixed 2 ns sampling period; support backend-defined sampling period
- Enable end-to-end experiment protocols including synchronized measurements
- Provide task-based, async-friendly new measurement primitive methods
- Add sweep measurement support in `measurement` layer

## Prioritized TODO

Legend: `P0` = highest, `P1` = important, `P2` = follow-up

| Priority | Task | Due | Dependency | Status |
| --- | --- | --- | --- | --- |
| P0 | Define QuEL-3 integration design (adapter boundary, lifecycle, error model) | TBD (after `quelware-client` completion) | `quelware-client` completion | PROPOSED / ON_HOLD |
| P0 | Implement QuEL-3 adapter with `quelware-client` | TBD (after `quelware-client` completion) | `quelware-client` completion | ON_HOLD (skeleton exists; implementation intentionally paused) |
| P0 | Prepare compatibility contract tests at `MeasurementClient` level (and `Experiment` facade delegation smoke checks) | 2026-02-25 | Existing controller APIs | IN_PROGRESS (factory-hook path covered) |
| P0 | Implement synchronized measurement protocol execution path | TBD (after `quelware-client` completion) | `quelware-client` completion + task primitives baseline | PROPOSED / ON_HOLD |
| P0 | Audit and remove fixed `2 ns` sampling assumptions in measurement/protocol path | 2026-02-26 | QuEL-3 timing model | IN_PROGRESS (measurement result time-axis path migrated) |
| P1 | Implement new task-based async measurement primitives | 2026-02-26 | Core task model decisions | TODO |
| P1 | Add sweep measurement API and execution in `measurement` layer | 2026-03-08 | Async primitives | TODO |
| P1 | Publish beta release notes and migration notes | 2026-02-28 | Major features for beta fixed | TODO |
| P1 | GA hardening: bug fixes from beta feedback | 2026-03-20 | Beta feedback | TODO |
| P1 | GA release notes and documentation finalization | 2026-03-25 | GA scope frozen | TODO |
| P1 | Finalize MkDocs user-guide and developer-guide for v1.5.0 (sufficient for experiment users and developers) | 2026-03-25 | GA scope frozen and config/runtime behavior stabilized | TODO |
| P2 | Developer ergonomics improvements (logs/errors/examples) for new flows | 2026-03-25 | Main features implemented | TODO |

Calendar note:

- `2026-02-29` does not exist; end-of-February deadlines are normalized to `2026-02-28`.

## Execution order (as of 2026-02-18, dependency-adjusted)

1. Wave A (current): non-QuEL-3 freeze and test scope lock
   - Lock compatibility and delegation contract scope.
   - Define synchronized scenario and beta release-note template.
   - Keep QuEL-3 decisions as `PROPOSED` until dependency completion.
2. Wave B (next): non-QuEL-3 beta-blocking implementation
   - Close remaining fixed `2 ns` assumptions in runtime and experiment paths.
   - Implement task-based async measurement primitives.
   - Add sweep measurement API and execution flow in measurement layer.
3. Wave C (after Wave B): beta gate for non-QuEL-3 scope
   - Run required checks: `uv run ruff check`, `uv run ruff format`, `uv run pyright`, `uv run pytest`.
   - Run hardware gate scenarios for QuEL-1.
   - Publish beta release notes + known limitations + migration notes.
4. Wave Q (triggered when `quelware-client` is ready): QuEL-3 track resume
   - Re-open and finalize DF-01 to DF-04.
   - Implement synchronized execution path and QuEL-3 runtime integration.
   - Run QuEL-3 hardware gates.
5. Wave D (2026-03-01 to 2026-03-25): GA hardening and release
   - Triage beta feedback and close critical/high issues.
   - Finalize sweep/async docs and compatibility notes.
   - Finalize GA release notes and MkDocs guides.

## Beta exit criteria (must pass)

- QuEL-3 basic control flow works on target environment
- Existing controller regression tests all pass
- QuEL-3 is API-compatible at `MeasurementClient` level
- `Experiment` core flows remain operational through delegation
- No blocking fixed `2 ns` assumptions remain in QuEL-3 code path
- Core synchronized protocol path is executable
- `mock_mode=True` compatibility path is covered by tests and remains operational
- Required tests are added and green (`uv run pytest`)
- Required quality checks are green (`uv run ruff check`, `uv run ruff format`, `uv run pyright`)
- Beta docs are available (how to run, known limitations)

## GA exit criteria (must pass)

- Beta issues triaged and critical/high issues closed
- QuEL-3 + existing controllers compatibility verified at `MeasurementClient` level
- Sampling-period differences are handled without API breakage
- Synchronized protocol and sweep measurement are documented and tested
- Migration/upgrade notes finalized
- Release notes finalized
- MkDocs `user-guide` and `developer-guide` are updated to final v1.5.0 behavior and provide sufficient end-to-end understanding for experiment users and developers

## Compatibility contract draft (`Experiment` / `MeasurementClient`)

### `Experiment` level (compatibility by delegation)

- `Experiment` is not treated as the primary compatibility contract surface for QuEL-3
- Keep delegation behavior operational for core flows:
  - `connect`, `disconnect`, `reload`, `run`
  - `execute`, `measure`, `measure_state`, `measure_idle_states`

### `MeasurementClient` level (must keep compatible for QuEL-3)

- Public alias compatibility (`Measurement` -> `MeasurementClient`)
- Constructor compatibility policy is practical/source-compatible:
  - required args and primary behavior must remain compatible
  - strict equality of all optional defaults is not required
- Lifecycle/API compatibility:
  - `load`, `connect`, `reload`, `disconnect`
  - `execute_measurement_schedule`, `execute`, `measure`
  - `create_measurement_config`, `build_measurement_schedule`
- Legacy delegation behavior from old measurement APIs remains compatible (keep delegation tests green)
- Measurement result compatibility criteria are type/shape centered
- Timing semantics must not assume fixed `2 ns`; schedule/config creation must work with backend-defined sampling period
- Canonical sampling period source is backend/controller `dt`

### Out of scope for compatibility guarantee (allowed to diverge)

- Backend/controller internals below `MeasurementClient`
- Device-specific execution internals for QuEL-3
- Internal adapter structure and lower protocol handling
- High-level `Experiment` contrib APIs (`measure_bell_state`, `measure_ghz_state`, etc.)

## Questions to finalize compatibility contract

- [x] `Experiment` compatibility excludes high-level contrib APIs; only core measurement/lifecycle delegation flows are in scope
- [x] Constructor compatibility policy: practical/source-compatible (required args + major behavior)
- [x] Measurement result compatibility policy: type/shape centered
- [x] `mock_mode=True` is mandatory for v1.5.0 beta compatibility
- [x] Canonical sampling period source of truth: backend/controller `dt`

## Current sprint checklist (2026-02-18 to 2026-02-21)

- [ ] Finalize QuEL-3 integration interface based on current `quelware-client` source (`quel3-adapter-interface-draft.md`) after dependency completion.
- [ ] Freeze decisions in `quel3-adapter-interface-draft.md` (alias mapping, capture key policy, trigger model, result shape) after dependency completion.
- [x] Define minimal synchronized protocol scenario for beta sign-off (`v1-5-0-synchronized-protocol-scenario.md`).
- [x] Lock `MeasurementClient` and `Experiment` delegation smoke test scope for beta (`v1-5-0-contract-test-scope.md`).
- [x] Draft beta release notes template and known limitation section (`v1-5-0-beta-release-notes-template.md`).
- [x] Document shared pulse factory design for backend-derived sampling-period ownership across `Experiment` and `MeasurementClient` (`v1-5-0-shared-pulse-factory-design.md`).
- [ ] Continue with non-QuEL-3 P1 implementation (`async primitives`, `sweep API`, remaining `2 ns` removals).

## Sampling-period audit (2026-02-17)

### P0: runtime/data-path blocks

- `src/qubex/backend/quel1/quel1_backend_constants.py`: `SAMPLING_PERIOD = 2.0` is globally fixed
- `src/qubex/measurement/measurement_schedule_builder.py`: capture start/duration are derived with fixed `SAMPLING_PERIOD`
- `src/qubex/measurement/adapters/backend_adapter.py`: schedule validation and capture-slot conversion depend on fixed `SAMPLING_PERIOD`
- `src/qubex/measurement/models/measure_result.py`: result time axis uses fixed single/avg sampling periods
- `src/qubex/experiment/experiment_constants.py`: experiment-wide sampling period aliases QuEL-1 fixed constant

### P1: experiment/contrib behavior using fixed period assumptions

- `src/qubex/experiment/services/calibration_service.py`: CR time-grid creation snaps to fixed `SAMPLING_PERIOD`
- `src/qubex/experiment/services/measurement_service.py`: state-evolution plotting uses fixed `SAMPLING_PERIOD`
- `src/qubex/experiment/services/characterization_service.py`: CPMG discretization uses `2 * SAMPLING_PERIOD`
- `src/qubex/contrib/simultaneous_coherence_measurement.py`: discretization uses `2 * SAMPLING_PERIOD`

### P2: visualization defaults

- `src/qubex/visualization/plotting.py`: `plot_waveform(..., sampling_period=2.0)` default

### Mapping rule to apply

- Replace fixed sampling-period usage in QuEL-3 path with backend/controller `dt`
- Keep QuEL-1 behavior unchanged by resolving `dt=2.0` through the same source-of-truth mechanism
- Keep sample-count based constants (`WORD_LENGTH`, `BLOCK_LENGTH`, etc.) and derive durations from `dt`

### Progress notes

- 2026-02-17: `measurement/models/measure_result.py` time-axis generation no longer hardcodes QuEL-1 fixed constants; backend-derived `sampling_period_ns` metadata is propagated through canonical result conversion.
- 2026-02-17: `avg` mode stride is now explicit metadata (`avg_sample_stride`, default `4`) to preserve 4-way multiplexed readout demodulation semantics while allowing backend-specific override.
- 2026-02-17: `MeasurementScheduleRunner.create_default()` now supports backend-provided custom adapter factory (`create_measurement_backend_adapter`) with contract tests, while keeping QuEL-1 defaults unchanged.
- 2026-02-17: Added `/docs/developer-notes/quel3-adapter-interface-draft.md` with `quelware-client` API mapping, constraint assumptions, and open questions for QuEL-3 adapter implementation.
- 2026-02-17: Added `MeasurementClient`-level contract test to ensure backend custom factory hooks are honored end-to-end by `execute_measurement_schedule()`.
- 2026-02-17: Added `Quel3MeasurementBackendAdapter` and `Quel3ExecutionPayload` skeleton for relaxed schedule validation and schedule-to-fixed-timeline payload conversion.
- 2026-02-17: Default adapter selection now supports explicit backend kind hint (`MEASUREMENT_BACKEND_KIND="quel3"`), with tests ensuring Quel3 adapter path selection.
- 2026-02-17: Added explicit runtime guard for `quel3` backend kind when backend executor hook is missing, to avoid accidental QuEL-1 executor fallback.
- 2026-02-17: `MeasurementScheduleRunner` now supports backend executors returning canonical `MeasurementResult` directly, enabling staged quelware integration before result-factory unification.
- 2026-02-17: Added built-in `Quel3BackendExecutor` (controller hook delegation) and tests for payload-type checks and hook presence checks.
- 2026-02-17: Added `instrument_aliases` in Quel3 payload with optional controller hook (`resolve_instrument_alias`) for target-to-alias mapping.
- 2026-02-17: Added explicit QuEL-1 capability hints on `Quel1BackendController` (`MEASUREMENT_BACKEND_KIND`, `MEASUREMENT_CONSTRAINT_MODE`, `MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE`).
- 2026-02-17: Added `Quel3BackendController` scaffold and session-scoped backend-family selection (`backend_kind`) in `SystemManager`/`MeasurementClient` (`quel1` or `quel3`, no mixed session).
- 2026-02-17: Implemented initial QuEL-3 backend execution path through `BackendController.execute(request)` that invokes quelware fixed-timeline execution and returns canonical `MeasurementResult` (with clear runtime error when quelware dependency is unavailable).
- 2026-02-18: Added `ConfigLoader` support for `wiring.v2.yaml` (`schema_version: 2`) including physical-id maps (`control`/`readout`) and port-label normalization (`p2tx`, `p0p1trx`) into runtime wiring rows.
- 2026-02-18: `SystemManager.load(..., backend_kind="quel3")` now prefers `wiring.v2.yaml` when present and falls back to legacy `wiring.yaml`.
- 2026-02-18: Started configuration-layer modularization by adding `qubex.configuration.wiring` and delegating wiring-v2 normalization from `ConfigLoader`.
- 2026-02-18: Moved `ConfigLoader` implementation to `qubex.configuration.config_loader`; kept `qubex.backend.config_loader` as compatibility shim and added import-compatibility tests.
- 2026-02-18: Added explicit `ConfigLoader.load()` lifecycle with `autoload` transition option; `SystemManager.load()` now uses explicit loader lifecycle (`autoload=False` + `load()`).
- 2026-02-18: `SystemManager.load(chip_id=...)` now resolves backend family from `chip.yaml` (`backend`) when argument is omitted, with precedence `explicit argument > chip.yaml > quel1 default`.
- 2026-02-18: Documented draft split of `chip.yaml`/`system.yaml` responsibilities (`topology.type` in `chip.yaml`, backend-specific runtime sections in `system.yaml`) in developer notes.
- 2026-02-18: `SystemManager.load(chip_id=...)` backend resolution now supports `system.yaml` (`backend`) with precedence `explicit argument > system.yaml > chip.yaml > quel1 default`.
- 2026-02-18: `ConfigLoader` now reads optional `system.yaml`; `ControlSystem.clock_master_address` prefers `system.yaml` `quel1.clock_master` and falls back to legacy `chip.yaml` `clock_master`.
- 2026-02-18: `ExperimentContext.register_custom_target()` now resolves qubits via `TargetRegistry` (or explicit `qubit_label`) and validates port/channel/target-type mapping before registration.
- 2026-02-18: Updated configuration docs/examples to align with current runtime behavior (`system.yaml` backend selection + `quel1.clock_master` support, QuEL-3 runtime endpoint/port/trigger using controller defaults in v1.5.0 pre-release).
- 2026-02-18: Reordered this release plan into execution waves (A to D), normalized invalid February end dates to `2026-02-28`, and added a current sprint checklist.
- 2026-02-18: Added QuEL-3 beta decision-freeze candidates (`DF-01` to `DF-04`) in `quel3-adapter-interface-draft.md`.
- 2026-02-18: Added `v1-5-0-synchronized-protocol-scenario.md` with `SP-BETA-001` as minimum synchronized-path beta gate scenario.
- 2026-02-18: Added `v1-5-0-beta-release-notes-template.md` and linked it from current sprint checklist.
- 2026-02-18: Added `v1-5-0-contract-test-scope.md` and fixed beta contract-test surface/gap list.
- 2026-02-18: Added facade delegation tests for `Experiment.run`, `connect`, `reload`, and `measure_idle_states` in `tests/experiment/test_experiment_facade_delegation.py`.
- 2026-02-18: Returned QuEL-3 decision items (`DF-01` to `DF-04`) to `PROPOSED` and deferred finalization until `quelware-client` completion.
- 2026-02-18: Switched execution order to prioritize non-QuEL-3 tasks while QuEL-3 dependency remains incomplete.
- 2026-02-18: Updated `ExperimentUtil.discretize_time_range()` to resolve sampling period from backend/controller (`DEFAULT_SAMPLING_PERIOD`) with QuEL-1 fallback, and added regression tests.
- 2026-02-18: Aligned experiment/contrib timing paths with backend-defined sampling period (`MeasurementClient.sampling_period`) and added `ExperimentContext` synchronization to apply backend dt to pulse-library sampling during init/connect/reload/configure.
- 2026-02-18: Added `v1-5-0-shared-pulse-factory-design.md` to define shared pulse construction architecture (backend/session-scoped, shared by `Experiment` and `MeasurementClient`); implementation is explicitly deferred to 2026-02-19 or later.

## Commit plan

- Commit 1 (today): planning and audit baseline
- Scope: release plan updates, compatibility contract finalization, sampling-period audit table
- Message draft: `docs: finalize v1.5.0 compatibility contract and sampling-period audit`

- Commit 2 (next): tests-first for compatibility and `dt` sourcing
- Scope: add/adjust tests for `MeasurementClient` compatibility, `mock_mode=True`, and `dt` propagation
- Message draft: `test: add MeasurementClient compatibility and dt-source contract coverage`

- Commit 3 (next): core runtime refactor to `dt`
- Scope: measurement schedule builder/adapter/result time axis and related constants wiring
- Message draft: `refactor(measurement): replace fixed sampling period with backend dt`

- Commit 4 (next): experiment/contrib follow-up and docs
- Scope: calibration/characterization/contrib timing discretization updates and release note deltas
- Message draft: `feat(experiment): align timing discretization with backend dt for QuEL-3`

## Backend constraint architecture (future-proof)

### Design goal

- Support both strict backends (QuEL-1 style) and relaxed backends (QuEL-3 style)
- Keep `MeasurementClient` API stable while backend-specific rules evolve

### Proposed model

- Introduce backend timing/constraint profile object provided by backend/controller:
  - `dt_ns` (required)
  - optional alignment constraints (`word_samples`, `block_samples`)
  - optional capture constraints (minimum gap, workaround-first-capture, etc.)
- Keep schedule construction mostly backend-agnostic
- Apply backend-specific validation and final quantization in adapter layer

### Policy by backend type

- Strict backend profile:
  - enforce word/block alignment and strict capture constraints
- Relaxed backend profile:
  - accept sample-grid placement based on `dt_ns`
  - delegate final packing/alignment to backend service (quelware)

### Implementation direction

- `MeasurementScheduleBuilder`: depend on `dt_ns` and generic profile inputs, not QuEL-1 constants
- backend adapters: own the strict/relaxed checks and conversion
- `MeasurementScheduleRunner.create_default()`: select adapter/profile based on backend/controller capability

## Development process updates

### Release gates

- Treat real-hardware validation as a release gate for beta and GA.
- Use `/Users/amachino/qubex/docs/developer-notes/hardware-validation-template.md` as the standard run sheet.

### Test layering

- PR-required checks:
  - `uv run ruff check`
  - `uv run ruff format`
  - `uv run pyright`
  - `uv run pytest`
- Hardware-integration checks:
  - Run on a scheduled basis and before release cut.
  - Cover both strict (QuEL-1) and relaxed (QuEL-3) backend profiles.

### Compatibility guardrails

- Changes on `MeasurementClient` compatibility surface must include contract-test updates.
- For profile/adapter changes, include strict and relaxed profile coverage in tests.

### Review DoD additions

- No new fixed `2 ns` assumption is introduced.
- Time discretization and alignment derive from backend/controller `dt`.
- Real-hardware validation results are logged with environment and scenario status.
