# Terminology

- **Chip**: The physical quantum device, identified by `chip_id` (for example, `64Q`).
- **Box**: A control or readout hardware unit defined in `box.yaml`.
- **Mux**: A multiplexed control/readout group on a chip.
- **Qubit**: A quantum element (for example, `Q00`).
- **Resonator**: Readout resonator target (for example, `RQ00`).
- **Target**: A qubit or resonator addressable by the control system.
- **Pulse**: A time-domain waveform applied to a target.
- **PulseSchedule**: An ordered, time-aware collection of pulses.
- **MeasurementConfig**: Execution settings such as `mode`, `shots`, `interval`, `frequencies`, and DSP flags/line parameters.
- **MeasurementResult / MeasureResult**: Structured results of a measurement, with plotting helpers.
- **ExperimentRecord**: Serialized experiment results for reproducibility.
- **ExperimentNote / CalibrationNote**: Key-value metadata stored alongside experiments.
- **StateClassifier**: Model (k-means or GMM) used for readout classification.
