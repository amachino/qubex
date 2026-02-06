# 1Q Benchmarking

This tutorial evaluates single-qubit gate quality using:

1. Standard randomized benchmarking (RB)
2. Interleaved randomized benchmarking (IRB)

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    muxes=[0],
)

exp.connect()
target = exp.qubit_labels[0]
```

## 2. Prepare calibrated DRAG pulses

```python
exp.calibrate_drag_hpi_pulse()
exp.calibrate_drag_pi_pulse()
```

## 3. Run RB and IRB

```python
rb = exp.randomized_benchmarking(
    target,
    n_cliffords_range=np.arange(0, 1001, 100),
    n_trials=30,
    save_image=True,
)

irb_x90 = exp.interleaved_randomized_benchmarking(
    target,
    interleaved_clifford="X90",
    n_cliffords_range=np.arange(0, 1001, 100),
    n_trials=30,
    save_image=True,
)
```

## Related example

- `docs/examples/experiment/5_randomized_benchmarking.ipynb`
