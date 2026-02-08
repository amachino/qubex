# 2Q calibration and benchmarking

This tutorial shows a common two-qubit flow for CR-based `ZX90`:

1. Obtain CR parameters
2. Calibrate `ZX90`
3. Validate with sequence execution and Bell-state measurement
4. Estimate gate performance with IRB

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[8, 10],
)

exp.connect()
control = "Q08"
target = "Q10"
pair = f"{control}-{target}"
```

## 2. Calibrate `ZX90`

```python
exp.obtain_cr_params(control, target)

exp.calibrate_zx90(
    control_qubit=control,
    target_qubit=target,
)
```

## 3. Validate pulse behavior

```python
zx90 = exp.zx90(control, target)
zx90.plot()

exp.repeat_sequence(zx90).plot()
exp.measure_bell_state(control, target).plot()
```

## 4. Run interleaved RB for `ZX90`

```python
irb_zx90 = exp.interleaved_randomized_benchmarking(
    pair,
    interleaved_clifford="ZX90",
    n_cliffords_range=np.arange(0, 21, 2),
    n_trials=30,
)
```

Update `params.yaml` / `props.yaml` as needed and run `exp.reload()` between calibration steps.

## Related example

- `docs/examples/experiment/9_cr_calibration.ipynb`
