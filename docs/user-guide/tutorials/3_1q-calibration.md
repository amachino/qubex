# 1Q Calibration

This tutorial calibrates key single-qubit parameters:

1. Control frequency
2. Readout frequency
3. `pi/2` and `pi` pulses

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    muxes=[0],
)

exp.connect()
targets = exp.qubit_labels
```

## 2. Run frequency calibrations

```python
exp.obtain_rabi_params()

exp.calibrate_control_frequency(
    targets,
    detuning_range=np.linspace(-0.01, 0.01, 21),
    time_range=range(0, 101, 4),
)

exp.calibrate_readout_frequency(
    targets,
    detuning_range=np.linspace(-0.01, 0.01, 21),
    time_range=range(0, 101, 4),
)
```

## 3. Run pulse calibrations

```python
exp.calibrate_hpi_pulse(targets, n_rotations=1)
exp.calibrate_pi_pulse(targets)
```

After updating parameter files, reload and validate:

```python
exp.reload()
exp.obtain_rabi_params()
exp.repeat_sequence(exp.hpi_pulse, repetitions=20).plot(normalize=True)
```

## Related examples

- `docs/examples/experiment/2_frequency_calibration.ipynb`
- `docs/examples/experiment/3_pulse_calibration.ipynb`
