# `PulseSchedule` の組み方

`PulseSchedule` は、Qubex でパルスレベル系列を表す共通コンテナです。
`Experiment` と `Simulator` の両方で使われる共有概念であり、measurement 側でも
パルス系列を capture や execution の要求に変換するときに登場します。

このページは、まずパルス系列の組み方を理解してから、
それを実機で流すかオフラインで使うかを決めたいときの入口です。

## `PulseSchedule` は何に使うか

- 1 つ以上の channel にまたがる時系列の pulse event を表現する
- 実機ワークフローとシミュレーションワークフローのあいだで同じ pulse object や schedule パターンを再利用する
- まず制御系列を組み立て、その後で実行方法や解析方法を選ぶ

## 最小パターン

schedule を作り、`with` ブロックの中で pulse を追加します。
次の block に進む前に channel を揃えたいときは `barrier()` を使います。

```python
import numpy as np
import qubex as qx

Q0 = "Q00"
Q1 = "Q01"

pulse = qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16)

schedule = qx.PulseSchedule()
with schedule as s:
    s.add(Q0, pulse)
    s.add(Q0, pulse.scaled(2.0))
    s.barrier()
    s.add(Q1, pulse.shifted(np.pi / 6))

schedule.plot()
```

## 基本の考え方

- `add(channel, pulse)`: 1 つの channel に pulse event を配置する
- `barrier()`: 次の block に入る前に channel を揃える
- 自動 padding: `with` ブロックを抜けると、各 channel は同じ長さに揃えられる
- pulse の再利用: `scaled()` や `shifted()` を使って、1 つの base pulse から派生 pulse を作れる

## 次にどこで使うか

- `Experiment`: 実機で流すワークフローには [クイックスタート](../getting-started/quickstart.md) から進む
- `Simulator`: 同じ pulse object と schedule の組み方を [Simulator サンプルワークフロー](../simulator/examples.md) で再利用する
- `低レベル API`: [Measurement API 概要](../measurement/index.md) から measurement 側のフローに変換する

## さらに学ぶ

- [Pulse tutorial notebook](../../examples/pulse/tutorial.ipynb)
- [Shape hash and waveform reuse](../../examples/pulse/shape_hash_and_waveform_reuse.ipynb)
- [サンプル集](../../examples/index.md)
