# パルスシーケンスの組み方

`PulseSchedule` は、Qubex でパルスシーケンスを表す共通コンテナです。
`Experiment` と `QuantumSimulator` の両方で使われる共有概念です。

## `PulseSchedule` は何に使うか

- 1 つ以上の channel にまたがる時系列の pulse event を表現する
- 実機ワークフローとシミュレーションワークフローのあいだで同じ `Pulse` オブジェクトや schedule パターンを再利用する
- まずパルスシーケンスを組み立て、その後で実行方法や解析方法を選ぶ

## 最小パターン

`PulseSchedule` インスタンスを作り、`with` ブロックの中で pulse を追加します。
次のブロックに進む前にチャンネルを揃えたいときは `barrier()` を使います。

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

## 使い方

- `add(channel, pulse)`: 1 つのチャンネルに pulse event を配置する
- `barrier()`: 次のブロックに入る前にチャンネルを揃える
- `barrier(labels=[...])`: 特定のチャンネルだけに barrier を適用する
- `call(schedule)`: 別の `PulseSchedule` オブジェクトをその場に挿入する
- 自動 padding: `with` ブロックを抜けると、各チャンネルは同じ長さに揃えられる
- `Pulse` の再利用: `scaled()` や `shifted()` を使って、1 つのベース `Pulse` から派生 `Pulse` を作れる

## このあと使う場面

- `Experiment`: 実機で流すワークフローには [クイックスタート](../getting-started/quickstart.md) から進む
- `QuantumSimulator`: 同じ `Pulse` オブジェクトと schedule の組み方を [QuantumSimulator サンプルワークフロー](../simulator/examples.md) で再利用する
- `低レベル API`: [低レベル API 概要](../low-level-apis/index.md) から、`measurement` 実行や backend 固有経路へ進む

## さらに学ぶ

- [Pulse tutorial notebook](../../examples/pulse/tutorial.ipynb)
- [Shape hash and waveform reuse](../../examples/pulse/shape_hash_and_waveform_reuse.ipynb)
- [サンプル集](../../examples/index.md)
