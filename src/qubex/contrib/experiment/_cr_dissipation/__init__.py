"""
Provide the private implementation of CR dissipation characterization.

Modules are organized by responsibility:

- `experiment`: validation, acquisition orchestration, plots, and result assembly.
- `pulses`: hardware schedules and matching semantic pulse descriptions.
- `analysis`: protocol fits, rate extraction, idle baselines, and fidelity limits.
- `fit`: generic GLS fitting and covariance helpers.
- `simulation`: two-qutrit channels and fidelity propagation.
- `types`: data records exchanged between the layers above.

Only `qubex.contrib.experiment.cr_dissipation` is a public API.
"""
