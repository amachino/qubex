"""
Provide the private implementation of CR dissipation characterization.

Modules are organized by responsibility and follow the execution flow:

- `experiment`: public-call validation, IX45 calibration, acquisition, conversion
  to typed measurements, plotting/reporting, and final `Result` assembly.
- `pulses`: calibrated hardware schedules, diagnostic references, and matching
  semantic operations used by the simulator.
- `analysis`: observable reconstruction, A/B and C/D fits, non-recursive idle
  baselines, rate/status assembly, and fidelity limits.
- `fit`: generic GLS fitting and covariance helpers.
- `simulation`: semantic two-qutrit channels, observables, leakage-aware
  fidelity, and sigma-point uncertainty propagation.
- `types`: immutable records exchanged between these layers.

Only `qubex.contrib.experiment.cr_dissipation` is a public API.
"""
