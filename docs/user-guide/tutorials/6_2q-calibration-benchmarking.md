# 2Q calibration and benchmarking

This tutorial walks through a common two-qubit flow for CR-based `ZX90`
calibration, validation, and benchmarking.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[8, 10],
)

exp.connect()
control, target = exp.qubit_labels[:2]
```

## 2. Obtain CR parameters

Start by estimating the CR parameters for the control-target pair.

```python
exp.obtain_cr_params(control, target)
```

## 3. Calibrate `ZX90`

```python
exp.calibrate_zx90(
    control_qubit=control,
    target_qubit=target,
)
```

## 4. Validate the calibrated gate

Use the calibrated pulse schedule directly and check a simple Bell-state
measurement.

```python
zx90 = exp.zx90(control, target)
zx90.plot()

exp.repeat_sequence(zx90).plot()
exp.measure_bell_state(control, target).plot()
```

## 5. Run interleaved RB for `ZX90`

```python
result_irb = exp.interleaved_randomized_benchmarking(
    f"{control}-{target}",
    interleaved_clifford="ZX90",
    n_cliffords_range=np.arange(0, 21, 2),
    n_trials=30,
)
```
