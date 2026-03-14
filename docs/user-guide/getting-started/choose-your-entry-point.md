# Choose your entry point

Qubex provides three main entry points. Choose the one that matches the level of abstraction and runtime environment you need.

## Quick comparison

| Entry point | Choose it when you want to | Start here |
| --- | --- | --- |
| `Experiment` | Run pulse-level qubit experiments through a high-level workflow on real hardware | [Experiment overview](../experiment/index.md) |
| `Measurement` | Control hardware-backed measurement sessions, custom schedules, and low-level readout flows directly | [Measurement overview](../measurement/index.md) |
| `Simulator` | Study pulse-level Hamiltonian dynamics offline without using real hardware | [Simulator overview](../simulator/index.md) |

## Shared prerequisites

- Install Qubex first: [Installation](installation.md)
- For `Experiment` and `Measurement`, prepare configuration and parameter files for your system: [System configuration](system-configuration.md)

## Experiment

Choose `Experiment` when you want Qubex to manage the common research workflow around connection, configuration, pulse construction, execution, and analysis.

Recommended path:

- [Experiment overview](../experiment/index.md)
- [Quickstart](quickstart.md)
- [Experiment example workflows](../experiment/examples.md)

## Measurement

Choose `Measurement` when you need more direct control over measurement sessions, backend execution, schedule construction, or readout-specific workflows on real hardware.

Recommended path:

- [Measurement overview](../measurement/index.md)
- [Measurement example workflows](../measurement/examples.md)

## Simulator

Choose `Simulator` when you want to iterate on pulse-level models, calibrations, and dynamics without connecting to hardware.

Recommended path:

- [Simulator overview](../simulator/index.md)
- [Simulator example workflows](../simulator/examples.md)
