# 1Q characterization

This tutorial runs standard single-qubit characterization procedures for T1,
T2 echo, and Ramsey experiments.

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

## 2. Run `T1`

Start by measuring energy relaxation.

```python
result_t1 = exp.t1_experiment(
    [Q0],
    time_range=np.geomspace(100, 100e3, 51),
)
```

## 3. Run `T2 echo` and `Ramsey`

Then measure dephasing and detuning-sensitive coherence.

```python
result_t2 = exp.t2_experiment(
    [Q0],
    time_range=np.geomspace(300, 100e3, 51),
)

result_ramsey = exp.ramsey_experiment(
    [Q0],
    time_range=np.arange(0, 10_001, 100),
    detuning=0.001,
)
```

## 4. Plot results

```python
result_t1.plot()
result_t2.plot()
result_ramsey.plot()
```
