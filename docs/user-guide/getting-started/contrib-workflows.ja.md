# コミュニティ提供ワークフロー

Qubex の追加ワークフローの一部は、`Experiment` のコアメソッドではなく、`qubex.contrib` 配下のコミュニティ提供関数として提供されています。

このページは主に既存ユーザー向けの移行メモです。古い notebook や script で `Experiment` の helper が見つからなくなった場合は、対応する contrib 関数を使い、最初の引数に `exp` を渡してください。

```python
from qubex import contrib
```

## 移動した API

古い notebook や script を更新するときは、次の対応表を使ってください。

| `exp` での旧呼び出し | 新しい contrib 関数 |
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
