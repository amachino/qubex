# Experiment

`Experiment` is the recommended starting point for most hardware-backed Qubex users.
It provides a high-level workflow for configuring systems, connecting to instruments, building pulse sequences, running measurements, and analyzing results.

## Who should use Experiment

- Researchers who want to run pulse-level experiments on qubits through a high-level API
- Users who want built-in workflows for characterization, calibration, and benchmarking
- Teams who prefer one consistent entry point from setup through analysis

## Recommended path

1. Install Qubex: [Installation](../getting-started/installation.md)
2. Prepare your configuration files: [System configuration](../getting-started/system-configuration.md)
3. Work through the basic workflow: [Quickstart](../getting-started/quickstart.md)
4. Continue with curated notebooks: [Experiment example workflows](examples.md)
5. Explore extra routines when needed: [Community-contributed workflows](../getting-started/contrib-workflows.md)

## Experimental async APIs

`Experiment` also exposes async-first methods:

- `run_measurement()`
- `run_sweep_measurement()`
- `run_ndsweep_measurement()`

Treat these as Experimental features. They are public, but the signature,
behavior, and result-handling details may change in future releases while the
async workflow is still settling.

Prefer the legacy synchronous methods (`measure()`, `execute()`,
`sweep_parameter()`) when API stability is the priority today.

## Choose Low-level APIs instead when

- You want to work directly with `MeasurementSchedule`, capture/readout, sweeps, or other `measurement`-module execution flows
- You want direct control over configuration loading, `ExperimentSystem`, or synchronization
- You are building backend integrations, controller-level execution paths, or QuEL-specific runtime tooling

See [Low-level APIs overview](../low-level-apis/index.md) for the `measurement`, `system`, and `backend` paths.
