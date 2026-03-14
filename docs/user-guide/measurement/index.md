# Measurement

`Measurement` is the lower-level entry point for hardware-backed execution.
Use it when you need direct control over measurement sessions, custom schedules, backend execution, and readout-oriented workflows.

## Who should use Measurement

- Advanced users who want to control measurement sessions directly on real hardware
- Developers who need lower-level access to backend-facing execution and schedule construction
- Users building custom readout, sweep, or sequencer workflows outside the higher-level `Experiment` facade

## What Measurement gives you

- Session lifecycle control such as loading configuration, connecting hardware, and checking backend state
- Direct execution of measurement schedules and custom readout placement
- Sweep builders and execution helpers for structured low-level measurements
- Readout classification utilities and backend-specific execution hooks

## Recommended path

1. Install Qubex: [Installation](../getting-started/installation.md)
2. Prepare your hardware configuration: [System configuration](../getting-started/system-configuration.md)
3. Start with curated notebooks: [Measurement example workflows](examples.md)

## Choose Experiment instead when

- You want a higher-level workflow centered on experiments rather than measurement internals
- You want built-in characterization, calibration, and benchmarking routines
- You prefer one facade for setup, execution, and analysis

See [Experiment overview](../experiment/index.md) for that path.
