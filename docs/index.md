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

## Documentation map

- [Installation](user-guide/getting-started/installation.md)
- [System configuration](user-guide/getting-started/system-configuration.md)
- [Choose your entry point](user-guide/getting-started/choose-your-entry-point.md)
- [Experiment guide](user-guide/experiment/index.md)
- [Measurement guide](user-guide/measurement/index.md)
- [Simulator guide](user-guide/simulator/index.md)
- [Examples](examples/index.md)
- [API reference](api-reference/qubex/index.md)
