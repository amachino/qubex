# QuantumSimulator

`QuantumSimulator` is the offline entry point for pulse-level Hamiltonian studies.
Use it when you want to model quantum systems, drive them with pulses, and iterate on experiments without connecting to real hardware.

## Who should use QuantumSimulator

- Researchers who want to study pulse-level dynamics without using hardware
- Users exploring model behavior before moving to a real system
- Teams prototyping calibrations and pulse designs offline

## What QuantumSimulator gives you

- Pulse-level Hamiltonian simulation for qubits, resonators, and coupled systems
- Reuse of Qubex pulse objects in offline studies
- A safe path for iterating on calibrations before hardware time is available

## Recommended path

1. Install Qubex: [Installation](../getting-started/installation.md)
2. Learn the shared pulse-sequence model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
3. Start with curated notebooks: [QuantumSimulator example workflows](examples.md)

You do not need hardware configuration files to begin with the simulator notebooks.

## Choose Experiment instead when

- You want to run experiments on real hardware
- You need measurement results and hardware-backed readout
- You want the higher-level workflow around connection, execution, and analysis

See [Experiment](../experiment/index.md) for that path.
