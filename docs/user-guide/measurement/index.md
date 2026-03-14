# Measurement

`Measurement` is the measurement-centric entry point for hardware-backed execution.
It is the measurement-side foundation that `Experiment` uses for session lifecycle and execution.
Choose it when you want to work directly with sessions, schedules, capture/readout, sweeps, and backend integration as first-class concepts, not because it is a strictly more capable alternative to `Experiment`.

This page sits under [Low-level APIs](../low-level-apis/index.md).

## Who should use Measurement

- Users who want measurement-side objects and contracts to be their main abstraction
- Developers building backend integrations, readout utilities, or measurement-specific tooling
- Users writing custom session, schedule, sweep, or sequencer flows without centering the broader experiment-analysis workflow

## What Measurement gives you

- A measurement-centric API surface for session lifecycle, schedule execution, capture/readout handling, and sweeps
- Direct access to measurement-side data models and helpers
- Readout classification utilities and backend-specific execution hooks
- The same measurement foundation that `Experiment` delegates to internally

## Recommended path

1. Install Qubex: [Installation](../getting-started/installation.md)
2. Prepare your hardware configuration: [System configuration](../getting-started/system-configuration.md)
3. Learn the shared pulse-sequence model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
4. Read the section overview: [Low-level APIs](../low-level-apis/index.md)
5. Start with curated notebooks: [Measurement example workflows](examples.md)

## Choose Experiment instead when

- You want the recommended user-facing workflow for most hardware-backed experiments
- You want built-in characterization, calibration, and benchmarking routines
- You prefer one facade for setup, execution, and analysis without centering measurement-side vocabulary

See [Experiment overview](../experiment/index.md) for that path.
