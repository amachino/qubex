# Overview

Qubex is designed around an `Experiment`-first workflow.

Most users can run setup, calibration, characterization, benchmarking, and persistence only through `Experiment`.

Qubex organizes this workflow into three layers:

1. **Configuration layer**: YAML files define chip topology, wiring, and control parameters.
2. **Experiment layer**: `Experiment` provides the main user-facing APIs.
3. **Analysis layer**: Fit, visualize, and persist results for reproducibility.

`Measurement` is available as a lower-level interface for advanced control needs.

## Typical workflow

1. **Prepare configuration files**: Place YAML files under your chip directory.
2. **Create an experiment**: Initialize `Experiment(chip_id=..., config_dir=..., params_dir=...)`.
3. **Run experiments**: Use `measure`, calibration, characterization, and benchmarking APIs from `Experiment`.
4. **Analyze and save**: Use `result.plot()`, `qubex.analysis`, and `result.save()`.

## Main interfaces

- **`Experiment`**: High-level entry point for common experiments and calibrations.
- **`Measurement`**: Lower-level interface for detailed schedule and execution control (advanced users).

Most users do not need to access internal runtime classes directly.

## Related subpackages

- **`qubex.pulse`**: Pulse definitions and helpers.
- **`qubex.analysis`**: Fitting and analysis utilities.
- **`qubex.visualization`**: Plotting and timeline visualization utilities.
- **`qubex.clifford`**: Clifford generation utilities for benchmarking.
- **`qubex.simulator`**: Simulation interfaces backed by `qxsimulator`.
- **`qubex.diagnostics`**: Chip inspection and diagnostic helpers.
