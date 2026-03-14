# Low-level APIs

This section is for developers who work on backend integration, readout
utilities, measurement-side tooling, or sequencer-oriented execution paths.

Most Qubex users should start with [Experiment](../experiment/index.md) on real
hardware or [Simulator](../simulator/index.md) for offline work.
Come here when you need measurement-side concepts such as sessions, schedules,
capture/readout, sweeps, or backend execution details to be your primary
abstraction.

## Start here

- [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
- [Measurement API overview](../measurement/index.md)
- [Measurement example workflows](../measurement/examples.md)

## Typical use cases

- Building or validating backend integrations
- Implementing readout-specific utilities or analysis helpers
- Developing custom schedule, sweep, or sequencer flows
- Working with measurement-side data models and execution contracts directly

## Choose Experiment instead when

- You want the recommended user-facing workflow for most hardware-backed experiments
- You want built-in characterization, calibration, and benchmarking routines
- You prefer one facade for setup, execution, and analysis

## Related documentation

- [Choose where to start](../getting-started/choose-where-to-start.md)
- [Examples](../../examples/index.md)
- [Developer guide](../../developer-guide/index.md)
