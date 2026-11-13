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

## Sweep plans and temporary settings

Backend planners decide which sweep values can execute under one configuration.
The shared `SweepSegment` holds original indices, values, native configuration,
an optional reference value and whether reconfiguration is required. A plan is
a tuple of segments with frequency values in GHz; backend-specific settings
retain their native format.

`SystemManager` delegates target resolution and model synchronization to its
existing backend-specific `SystemSynchronizer`. Backend code owns frequency
limits, oscillator quantization, configuration application and restoration.
Experiment owns acquisition order, pulses, amplitudes and analysis. Measurement
executes schedules; it does not monitor configuration scopes.

```python
segments = system.plan_frequency_sweep(readout_target, frequencies)
for segment in segments:
    with system.modified_sweep_configuration(segment.configuration):
        for frequency in segment.values:
            with system.modified_frequencies({readout_target: frequency}):
                schedule = make_schedule(frequency)
                measurement.execute(schedule=schedule, shot_averaging=True)
```

Frequencies are in GHz. Backend constraints determine segmentation;
`subrange_width` limits the span of all points, including an overlap reference.
The planner takes the longest input prefix within this width and shortens it
until a backend configuration covers every point. Windows include both endpoints;
points are not balanced into equally sized groups. Input order and
duplicate points are retained. With `overlap=True`, subsequent segments carry
the preceding segment's last value as `reference_value`. Planning verifies that
this reference also fits both the requested width and the backend window; the
caller decides how to acquire it. References remain separate from measurement
values and indices. If a reference cannot be grouped with at least one new
point, planning raises `ValueError` instead of repeating the same position.

QuEL-3 selects the deployment span from the instrument's port, independently of
its role. It reserves the default 0.1 GHz margin on each side of every carrier,
including overlap references:

| Port | Deployment span | Usable sweep span |
| --- | --- | --- |
| `tx_*` | 1.8 GHz | 1.6 GHz |
| `trx_*` | 0.8 GHz | 0.6 GHz |

`subrange_width` adds an upper bound on the usable span. For example, a 0.9 GHz
limit on a TRX port still produces segments spanning at most 0.6 GHz. Both
`resonator_spectroscopy()` and `qubit_spectroscopy()` accept this optional
keyword and pass it to each power's frequency scan. Omitting it, or passing
None, retains the corresponding scan's existing default width limit.

During a scope, the planned values and references are realizable under the
applied settings. `modified_frequencies()` retains its calling convention.
QuEL-1 still updates and restores logical and registered target frequencies
without retuning LO/CNCO/FNCO.

QuEL-3 changes its restoration behavior: the scope saves the entry instrument
definitions, permits automatic deployment when a requested carrier requires it,
and restores only definitions changed during entry. It restores logical target
frequencies separately, without using those values to recalculate instrument
ranges. Planned carriers fit the prepared range with their margins, so nested
frequency scopes perform no deployment on entry or exit, even when the original
logical carrier lies outside the sweep window.

Explicit settings changes and resets remain the caller's responsibility.
`modified_backend_settings`, reset and `sweep_parameter` retain their behavior.
Automatic electrical-delay calibration inside resonator spectroscopy does not
prompt; direct `measure_electrical_delay()` calls retain confirmation by default.
When its start frequency is omitted, electrical delay uses `target.fine_frequency`
and falls back to `target.frequency` if that lookup raises `ValueError`.

Normal exit, acquisition errors and partial application failures restore touched
entry settings and corresponding model/cache state. Restoration errors propagate;
failed QuEL-1 restoration invalidates the affected box cache. Scopes do not
restore waveform memory or results and do not guarantee phase continuity or
arbitrary waveform spectral fidelity. Resonator spectroscopy retains its existing power-first orchestration and
`[power, frequency]` results. Replan after changing wiring or other planning assumptions.
