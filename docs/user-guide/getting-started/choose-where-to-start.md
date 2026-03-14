# Choose where to start

Getting started with Qubex is a two-step process: prepare the environment, then
follow the entry point that matches your goal.
This page lays out that flow in the same order you will actually use it.

## Prepare these first

- Every user should start with [Installation](installation.md).
- Hardware-backed `Experiment` workflows also require [System configuration](system-configuration.md).
- `QuantumSimulator` does not require hardware configuration files.

## Choose `Experiment` for hardware-backed experiments

This is the recommended entry point when you want a high-level workflow for real
hardware experiments.
Use it for connection, measurement execution, characterization, calibration,
benchmarking, and result analysis in one path.

Recommended path:

1. Start with [`Experiment`](../experiment/index.md)
2. Work through [Quickstart](quickstart.md)
3. Continue with [Experiment example workflows](../experiment/examples.md)
4. Use [Community-contributed workflows](contrib-workflows.md) when needed

## Choose `QuantumSimulator` for offline studies

This is the entry point when you want to study pulse-level dynamics or iterate
on pulse design without connecting to hardware.
Use it for offline modeling and trial-and-error before moving to a real system.

Recommended path:

1. Start with [`QuantumSimulator`](../simulator/index.md)
2. Learn the shared model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
3. Start with [QuantumSimulator example workflows](../simulator/examples.md)

## Choose Low-level APIs for `measurement`, `system`, or `backend` work

This is the entry point when you want to work below the `Experiment` facade.
The section is organized around three modules:

- [`measurement`](../measurement/index.md): `MeasurementSchedule`, capture/readout, sweeps, and measurement execution flows
- [`system`](../system/index.md): configuration loading, system models, and synchronization
- [`backend`](../backend/index.md): backend controllers, execution requests, and QuEL-specific runtimes

Recommended path:

1. Start with [Low-level APIs overview](../low-level-apis/index.md)
2. Learn the shared model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
3. Choose the module page that matches the abstraction you need
4. Continue with the matching example workflows: [`measurement`](../measurement/examples.md), [`system`](../system/examples.md), or [`backend`](../backend/examples.md)

## Supporting pages

- [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md): use this when you want to understand the shared `PulseSchedule` model used across `Experiment`, `QuantumSimulator`, and the low-level execution stack.
- [Examples](../../examples/index.md): use this when you want to browse notebooks by topic instead of by workflow.

## Contributing instead of using Qubex

If you want to extend Qubex or work on the codebase itself, start with [Contributing](../../CONTRIBUTING.md) and then continue with the [Developer guide](../../developer-guide/index.md).
