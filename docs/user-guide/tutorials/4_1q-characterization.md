# 1Q Characterization

This tutorial measures coherence and detuning indicators using:

1. T1
2. T2 echo
3. Ramsey

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0],
)

exp.connect()
q = exp.qubit_labels[0]
```

## 2. Run characterization experiments

```python
result_t1 = exp.t1_experiment(
    [q],
    time_range=np.logspace(np.log10(100), np.log10(100 * 1000), 51),
)

result_t2 = exp.t2_experiment(
    [q],
    time_range=np.logspace(np.log10(300), np.log10(100 * 1000), 51),
)

result_ramsey = exp.ramsey_experiment(
    [q],
    time_range=np.arange(0, 10_001, 100),
    detuning=0.001,
)
```

## 3. Plot results

```python
result_t1.plot()
result_t2.plot()
result_ramsey.plot()
```

## Related example

- `docs/examples/experiment/4_t1_t2_experiments.ipynb`
