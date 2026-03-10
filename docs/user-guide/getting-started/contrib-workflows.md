# Community-Contributed Workflows

Some additional workflows in Qubex are provided as community-contributed
functions under `qubex.contrib` rather than as core `Experiment` methods.

This page is mainly a migration note for existing users. If an older notebook
or script calls an `Experiment` helper that is no longer available there, use
the corresponding contrib function instead and pass `exp` as the first
argument.

```python
from qubex import contrib
```

## Moved APIs

Use this mapping when updating older notebooks or scripts:

| Old call on `exp` | New contrib function |
| --- | --- |
| `exp.measure_cr_crosstalk(...)` | `contrib.measure_cr_crosstalk(exp, ...)` |
| `exp.cr_crosstalk_hamiltonian_tomography(...)` | `contrib.cr_crosstalk_hamiltonian_tomography(exp, ...)` |
| `exp._simultaneous_measurement_coherence(...)` | `contrib.simultaneous_coherence_measurement(exp, ...)` |
| `exp._stark_t1_experiment(...)` | `contrib.stark_t1_experiment(exp, ...)` |
| `exp._stark_ramsey_experiment(...)` | `contrib.stark_ramsey_experiment(exp, ...)` |
| `exp.purity_benchmarking(...)` | `contrib.purity_benchmarking(exp, ...)` |
| `exp.interleaved_purity_benchmarking(...)` | `contrib.interleaved_purity_benchmarking(exp, ...)` |

## Simultaneous coherence

```python
import numpy as np
from qubex import contrib

results = contrib.simultaneous_coherence_measurement(
    exp,
    targets=[Q0, Q1],
    time_range=np.arange(0, 20_001, 1000),
    n_shots=1024,
)

t1_result = results["T1"]
t1_result.plot()
```

## Stark-driven characterization

```python
from qubex import contrib

stark_result = contrib.stark_t1_experiment(
    exp,
    targets=[Q0],
    stark_detuning=0.05,
    stark_amplitude=0.1,
    n_shots=1024,
)

stark_result.plot()
```

## Purity benchmarking

```python
from qubex import contrib

pb_result = contrib.purity_benchmarking(
    exp,
    targets=[Q0],
    n_shots=1024,
)

print(pb_result)
```
