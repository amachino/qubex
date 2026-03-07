# 1Q calibration

This tutorial walks through a typical single-qubit calibration flow for
frequencies and control pulses.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0, 1],
)

exp.connect()
```

## 2. Obtain Rabi parameters

Use current Rabi parameters as the starting point for pulse calibration and
validation.

```python
exp.obtain_rabi_params(exp.qubit_labels)
```

## 3. Calibrate control and readout frequencies

Run the frequency calibrations after the environment checks are complete.

```python
exp.calibrate_control_frequency(
    exp.qubit_labels,
    detuning_range=np.linspace(-0.01, 0.01, 21),
    time_range=np.arange(0, 101, 4),
)

exp.calibrate_readout_frequency(
    exp.qubit_labels,
    detuning_range=np.linspace(-0.01, 0.01, 21),
    time_range=np.arange(0, 101, 4),
)
```

## 4. Calibrate `pi/2` and `pi` pulses

After the frequencies are updated, calibrate the standard single-qubit control
pulses.

```python
exp.calibrate_hpi_pulse(exp.qubit_labels, n_rotations=1)
exp.calibrate_pi_pulse(exp.qubit_labels)
```

## 5. Reload and validate

If you update the parameter files during calibration, reload the experiment and
validate the calibrated pulses.

```python
exp.reload()
exp.obtain_rabi_params(exp.qubit_labels)

result = exp.repeat_sequence(exp.hpi_pulse, repetitions=20)
result.plot(normalize=True)
```
