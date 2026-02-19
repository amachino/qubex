# Overview

Qubex organizes experiment workflows into three layers:

1. **Configuration layer**: YAML files define chip topology, wiring, and control parameters.
2. **Execution layer**: `MeasurementClient` and `Experiment` translate pulses and schedules into device commands.
3. **Analysis layer**: Fit, visualize, and persist results for reproducibility.

## Typical workflow

1. **Load configuration**: `ConfigLoader` builds a `SystemManager` and `ExperimentSystem` from YAML files.
2. **Build pulses**: Use `qubex.pulse` or `qxpulse` to create waveforms and schedules.
3. **Execute measurements**: Run schedules via `MeasurementClient` or the `Experiment` facade.
4. **Analyze and save**: Use `result.plot()`, `qubex.analysis`, and `result.save()`.

## Main interfaces

- **`Experiment`**: High-level entry point for common experiments and calibrations.
- **`MeasurementClient`**: Detailed control over measurement schedules and execution.
- **`SystemManager`**: Singleton that caches configuration, backend controllers, and device state.
- **`ConfigLoader`**: Loads configuration and parameter files into a structured system model.

## Related subpackages

- **`qubex.pulse`**: Pulse definitions and helpers.
- **`qubex.analysis`**: Fitting and analysis utilities.
- **`qubex.visualization`**: Plotting and timeline visualization utilities.
- **`qubex.clifford`**: Clifford generation utilities for benchmarking.
- **`qubex.simulator`**: Simulation interfaces backed by `qxsimulator`.
- **`qubex.diagnostics`**: Chip inspection and diagnostic helpers.
