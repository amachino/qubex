# Qubex

Qubex is a qubit-control experimentation framework that unifies pulse-level experimental workflows on top of [QuEL](https://quel-inc.com/) control hardware. It supports everything from device configuration and arbitrary pulse-sequence execution to quantum-device characterization, quantum-gate calibration, benchmarking, and simulation experiments with a pulse-level simulator.

## Key features

- **End-to-end workflow**: Handle experimental setup, pulse-sequence execution, and result analysis in a single, consistent flow.
- **Backend-integrated setup**: Provide configuration files and let Qubex handle the setup for qubit-control experiments built from multiple QuEL hardware models.
- **Pulse-level control**: Run highly flexible experiments with arbitrary pulse sequences beyond circuit-level limits.
- **Standardized experiment routines**: Standardize workflows for quantum-device characterization, calibration, and benchmarking.
- **Pulse-level simulator**: Simulate the same pulse sequences as experiments at the Hamiltonian level.

## Start here

- [Installation](user-guide/getting-started/installation.md)
- [Choose where to start](user-guide/getting-started/choose-where-to-start.md)
- [System configuration](user-guide/getting-started/system-configuration.md)
- [Build pulse sequences with PulseSchedule](user-guide/pulse-sequences/index.md)

## Recommended paths

- [`Experiment`](user-guide/experiment/index.md): Recommended high-level entry point for most hardware-backed experiments.
- [`QuantumSimulator`](user-guide/simulator/index.md): Entry point for studying pulse-level Hamiltonian dynamics without real hardware.

## Low-level APIs

- [Overview](user-guide/low-level-apis/index.md): Start here to understand how `measurement`, `system`, and `backend` divide responsibility.
- [`measurement` module](user-guide/measurement/index.md): Work directly with `MeasurementSchedule`, capture/readout, sweeps, and measurement execution flows.
- [`system` module](user-guide/system/index.md): Load configuration, inspect `ExperimentSystem`, and synchronize runtime state.
- [`backend` module](user-guide/backend/index.md): Work with backend controllers, execution requests, and QuEL-specific implementations.

## Explore examples and APIs

- [Examples](examples/index.md)
- [Release notes](release-notes/index.md)
- [API reference](api-reference/qubex/index.md)

## Contribute to Qubex

- [Contributing](CONTRIBUTING.md)
- [Developer guide](developer-guide/index.md)
