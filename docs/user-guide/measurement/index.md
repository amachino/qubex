# `measurement` module

`qubex.measurement` is the measurement-centric module for hardware-backed
execution. It sits between [`system`](../system/index.md) and
[`backend`](../backend/index.md): it consumes loaded system state and turns
schedules into capture/readout and sweep execution flows.

This page sits under [Low-level APIs](../low-level-apis/index.md).

## Use `measurement` when

- Sessions, schedules, capture/readout, sweeps, and measurement results should be first-class concepts
- You want to work with `Measurement`, `MeasurementSchedule`, or sweep executors directly
- You want backend-neutral execution flows before dropping to backend-specific controllers

## Key objects

- `Measurement`: facade for session lifecycle and execution
- `MeasurementSchedule`, `MeasurementResult`, and sweep result models: canonical measurement-side contracts
- Builders and executors: `MeasurementScheduleBuilder`, `SweepMeasurementBuilder`, and `SweepMeasurementExecutor`
- Services and adapters: session/execution/classification services plus `MeasurementBackendAdapter` implementations

## Relationship to the other modules

- [`system`](../system/index.md): provides `ConfigLoader`, `ExperimentSystem`,
  targets, and parameter state that `Measurement` depends on
- [`backend`](../backend/index.md): provides the controller contracts and
  concrete QuEL-1/QuEL-3 runtimes that measurement adapters target

## Recommended path

1. Read the section overview: [Low-level APIs](../low-level-apis/index.md)
2. Learn the shared pulse-sequence model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
3. Start with curated notebooks: [`measurement` example workflows](examples.md)
4. Move to [`system`](../system/index.md) when configuration or synchronization is the main issue
5. Move to [`backend`](../backend/index.md) when controller-level payloads or execution paths are the main issue

## Choose `Experiment` instead when

- You want the recommended user-facing workflow for most hardware-backed experiments
- You want built-in characterization, calibration, and benchmarking routines
- You prefer one facade for setup, execution, and analysis without centering measurement-side vocabulary

See [`Experiment`](../experiment/index.md) for that path.
