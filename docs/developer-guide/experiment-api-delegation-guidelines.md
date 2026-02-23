# Experiment API delegation guidelines

This page defines the recommended delegation pattern for `Experiment` APIs.

## Purpose

- Keep `Experiment` as a stable user-facing facade.
- Keep domain logic in service/context layers.
- Avoid coupling `Experiment` directly to measurement internal services.

## Core rule

Use owner-based delegation, not a single mandatory chain.

- Measurement session/execution ownership:
  `Experiment` -> `SessionService` -> `Measurement` -> `Measurement*Service` (internal)
- Configuration synchronization ownership:
  `Experiment` -> `SessionService` -> `SystemManager`

`Experiment` should not directly depend on measurement internal service classes
such as `MeasurementSessionService` or `MeasurementExecutionService`.

## Recommended ownership by API category

| `Experiment` API category | Primary delegate | Notes |
| --- | --- | --- |
| environment/session (`connect`, `reload`, `check_status`, `disconnect`) | `SessionService` | Session flow delegates to measurement facade. |
| configuration sync (`configure`) | `SessionService` / `SystemManager` | Keep config state ownership in `SystemManager`. |
| context/read-only environment (`print_environment`, `print_boxes`, labels/registries/properties) | `ExperimentContext` | Keep experiment metadata/context ownership in context. |
| pulse construction/manipulation | `PulseService` | Keep pulse defaults and composition logic in pulse service. |
| measurement routines (`execute`, `measure`, tomography, classifier workflows) | `MeasurementService` | `MeasurementService` may use `ctx.measurement` internally. |
| calibration routines | `CalibrationService` | Keep calibration orchestration outside facade. |
| characterization routines | `CharacterizationService` | Keep analysis flow outside facade. |
| benchmarking routines | `BenchmarkingService` | Keep sequence/fit orchestration outside facade. |
| cross-service optimization workflows | `OptimizationService` | Keep multi-service composition outside facade. |

## Practical guidance

- Sampling-period synchronization ownership: `SessionService` owns pulse-library sampling period synchronization and executes it during initialization and session/configuration reload paths.

### Add a new `Experiment` API

1. Decide true owner (`ExperimentContext` or one of `*Service` classes).
2. Implement logic in the owner.
3. Add a thin forwarding method on `Experiment`.
4. Add/extend delegation tests in `tests/experiment/test_experiment_facade_delegation.py`.
5. Avoid duplicating default-value normalization in both facade and delegate.

### Measurement-side additions needed by `Experiment`

If `Experiment` needs a new measurement capability:

1. Add the capability to `Measurement` facade first.
2. Let `Measurement` route internally to `Measurement*Service`.
3. Call `ctx.measurement.<new_api>(...)` (or through `MeasurementService`) from experiment layer.

Do not bypass `Measurement` by importing measurement internal services in experiment layer.

### Configuration-side additions needed by `Experiment`

If a new API changes/pulls/pushes configuration state:

1. Prefer `SystemManager` operations from experiment side.
2. Keep backend session/control flow in measurement side.
3. Avoid mixing config sync logic into measurement execution APIs.

## Anti-patterns

- Calling measurement internal services directly from `Experiment` (or adding hard dependencies to those classes).
- Calling backend-controller methods directly from `Experiment` for behavior already owned by measurement/session services.
- Implementing non-trivial branching/business logic in `Experiment` facade methods.
- Letting `Experiment` depend on measurement package internals not required by its public contract.

## Current code shape (reference)

- `Experiment` delegates to `ExperimentContext` and domain services.
- `Experiment` delegates session lifecycle APIs directly to `SessionService`.
- `Experiment` instantiates and owns `SessionService`.
- `ExperimentContext` holds runtime context and `Measurement`.
- `Measurement` delegates lifecycle/execution internals to `Measurement*Service` classes.

This structure should remain the default unless an explicit migration design says otherwise.

## Related documents

- [Measurement backend responsibility policy](../developer-notes/measurement-backend-responsibility-policy.md)
- [v1.5.0 contract test scope](../developer-notes/v1-5-0-contract-test-scope.md)
