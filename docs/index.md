# Qubex

Qubex is a unified pulse-level framework powered by [QuEL](https://quel-inc.com/) control hardware, streamlining device configuration, arbitrary pulse-sequence execution, characterization, calibration, benchmarking, and offline experimentation with a pulse-level simulator.

## Key features

- **End-to-end workflow**: Handle experimental setup, pulse-sequence execution, and result analysis in a single, consistent flow.
- **Backend-integrated setup**: Provide config files and let Qubex handle the setup for qubit-control experiments across multiple QuEL hardware models.
- **Pulse-level control**: Run highly flexible experiments with arbitrary pulse sequences beyond circuit-level limits.
- **Automated experiment routines**: Standardize characterization, calibration, and benchmarking workflows.
- **Pulse-level simulator**: Simulate the same pulse sequences as experiments at the Hamiltonian level.

## Start here

- [Installation](user-guide/getting-started/installation.md)
- [Choose where to start](user-guide/getting-started/choose-where-to-start.md)
- [System configuration](user-guide/getting-started/system-configuration.md)
- [Build pulse sequences with PulseSchedule](user-guide/pulse-sequences/index.md)

## Recommended paths

- [Experiment](user-guide/experiment/index.md): Use the recommended user-facing workflow for most hardware-backed experiments.
- [QuantumSimulator](user-guide/simulator/index.md): Study pulse-level Hamiltonian dynamics offline without using real hardware.

## Low-level APIs

- [Overview](user-guide/low-level-apis/index.md): Start here when measurement-side abstractions are your main concern.
- [Measurement API overview](user-guide/measurement/index.md): Work directly with sessions, schedules, capture/readout, sweeps, and backend integration.

## Explore examples and APIs

- [Examples](examples/index.md)
- [API reference](https://amachino.github.io/qubex/api-reference/qubex/)

## Contribute to Qubex

- [Contributing](CONTRIBUTING.md)
- [Developer guide](developer-guide/index.md)
