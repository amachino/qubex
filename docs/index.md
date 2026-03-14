# Qubex

Qubex brings pulse-level workflows into a single framework for configuring heterogeneous devices, building and running arbitrary pulse sequences, applying standard characterization, calibration, and benchmarking routines, and iterating offline with a Pulse-level simulator.

The recommended user workflow is `Experiment`-first.
Lower-level `Measurement` APIs are available for advanced control.

## Key features

- **End-to-end workflow**: Go from setup to execution to analysis in a single, consistent flow.
- **Automated multi-hardware setup**: Define devices in config files and let Qubex set up mixed control stacks automatically.
- **Pulse-level control**: Run highly flexible experiments with arbitrary pulse sequences beyond circuit-level limits.
- **Automated experiment routines**: Standardize characterization, calibration, and benchmarking workflows.
- **Pulse-level simulator**: Simulate the same pulse sequences as experiments at the Hamiltonian level.

## Start here

- [Installation](user-guide/getting-started/installation.md)
- [System configuration](user-guide/getting-started/system-configuration.md)
- [Choose your entry point](user-guide/getting-started/choose-your-entry-point.md)

## Choose your workflow

- [Experiment guide](user-guide/experiment/index.md): Use the recommended high-level workflow for hardware-backed experiments.
- [Measurement guide](user-guide/measurement/index.md): Work directly with measurement sessions, custom schedules, and lower-level readout flows.
- [Simulator guide](user-guide/simulator/index.md): Study pulse-level Hamiltonian dynamics offline without using real hardware.

## Explore examples and APIs

- [Examples](examples/index.md)
- [API reference](https://amachino.github.io/qubex/api-reference/qubex/)

## Contribute to Qubex

- [Contributing](CONTRIBUTING.md)
- [Developer guide](developer-guide/index.md)
