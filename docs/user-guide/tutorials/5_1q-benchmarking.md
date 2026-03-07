# 1Q benchmarking

This tutorial evaluates calibrated single-qubit gates with randomized
benchmarking.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0],
)

exp.connect()
Q0 = exp.qubit_labels[0]
```

## 2. Calibrate DRAG pulses

Calibrate the DRAG `pi/2` and `pi` pulses before running benchmarking.

```python
exp.calibrate_drag_hpi_pulse(Q0)
exp.calibrate_drag_pi_pulse(Q0)
```

## 3. Run standard RB

```python
result_rb = exp.randomized_benchmarking(
    Q0,
    n_cliffords_range=np.arange(0, 1001, 100),
    n_trials=30,
    save_image=True,
)
```

## 4. Run interleaved RB

Use interleaved RB to estimate the quality of a specific calibrated gate.

```python
result_irb = exp.interleaved_randomized_benchmarking(
    Q0,
    interleaved_clifford="X90",
    n_cliffords_range=np.arange(0, 1001, 100),
    n_trials=30,
    save_image=True,
)
```
