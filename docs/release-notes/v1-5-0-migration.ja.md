# v1.5.0 移行ガイド

このガイドは `v1.4.8` から `v1.5.0` へ更新するユーザー向けです。
変更の全体像は [v1.5.0 リリースノート](v1-5-0.md) を参照してください。

## このガイドを読むべきユーザー

次のいずれかに当てはまる場合は読んでください。

- `Experiment` または `Measurement` で実機を動かしている
- `box.yaml`、`chip.yaml`、`wiring.yaml` などの設定ファイルを管理している
- `qubex.backend` から低レベル型を import している
- RZX、multipartite entanglement、purity benchmarking、Stark 系など、
  contrib 寄りの `Experiment` helper を使っている
- 固定 `2 ns` を前提にした timing-sensitive な code を持っている

一方、top-level の `qubex` import と QuEL-1 の基本的な
`Experiment.measure()` / `execute()` だけを使っており、移動した helper API
や backend import に依存していない場合は、比較的穏やかな更新で済みます。

## まず確認するチェックリスト

- Python `3.10+` を使う
- `chip_id` より `system_id` を優先する
- `system.yaml` を追加または見直す
- system 側の import を `qubex.backend` から `qubex.system` へ移す
- `shots` を `n_shots` に、`interval` を `shot_interval` に変える
- 移動した `Experiment` helper を `qubex.contrib` 呼び出しへ置き換える
- sweep、plot、timing utility で固定 `2 ns` を使わないようにする

## インストールと実行環境の変更

`v1.5.0` の repository workflow は `uv` 管理環境を前提にしています。
具体的な導入手順は
[インストールガイド](../user-guide/getting-started/installation.md) を参照してください。

最低限、次の前提は更新してください。

- Python `3.9` は非対応になりました。Python `3.10` 以上を使ってください。
- 実機向け依存関係は `backend` extra で導入します。
- repository 内開発は `uv` 環境で `make sync` を前提にしています。

## 設定変更

### `chip_id` 中心の読み込みから `system_id` 中心へ移行する

`v1.4.8` では実質的に single-chip 前提のワークフローが多くありました。
`v1.5.0` では、1 つの実行可能な装置構成を表す `system_id` が公開 API 上の
正規入口です。

旧スタイル:

```python
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/64Q/config",
    params_dir="/path/to/64Q/params",
)
```

新スタイル:

```python
import qubex as qx

exp = qx.Experiment(
    system_id="64Q-HF-Q1",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/64Q-HF-Q1",
)
```

`chip_id` は `v1.5.0` でも互換入力として使えますが、deprecated です。
更新する notebook では長期運用を見据えて `system_id` へ寄せてください。

### `system.yaml` を追加する

`system.yaml` は、1 つの runnable system と backend family を定義する
正規のファイルになりました。

```yaml
64Q-HF-Q1:
  chip_id: 64Q
  backend: quel1

144Q-LF-Q3:
  chip_id: 144Q
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
```

設定更新時のルール:

- `wiring.yaml` は chip 名だけでなく `system_id` で引く
- backend 選択は `system.yaml` に置く
- `config_dir` は共有 config directory、`params_dir` は system 単位の
  parameter directory として扱う

backend は次の優先順で決まります。

1. 明示的な `backend_kind` 引数
2. `system.yaml` の `backend`
3. 既定値 `quel1`

もし `chip.yaml` 側に backend 選択を入れていた場合は、
`system.yaml` に移してください。`v1.5.0` では `system.yaml` が存在する場合、
`chip.yaml` は backend の正規ソースではありません。

### structured parameter file を優先する

`v1.5.0` では parameter family ごとの structured YAML を優先します。

```yaml
meta:
  unit: GHz
  description: Example control frequencies
data:
  0: 5.000
  1: 5.125
```

推奨レイアウト:

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
  params/
    64Q-HF-Q1/
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
```

`params.yaml` と `props.yaml` は `v1.5.0` でも fallback input として読まれます。
一度に全部移行する必要はありませんが、新規更新分は per-file 形式へ寄せるのが
推奨です。

## API と import の変更

### system 側の import を `qubex.backend` から外す

最も大きい low-level import の変更は、system / configuration 系の型が
`qubex.backend` ではなくなったことです。

次のように書き換えてください。

```python
# v1.4.8
from qubex.backend import ConfigLoader, ControlSystem, ExperimentSystem, SystemManager

# v1.5.0
from qubex.system import ConfigLoader, ControlSystem, ExperimentSystem, SystemManager
```

`qubex.backend` のトップレベルは、backend controller contract と
`qubex.backend.quel1`、`qubex.backend.quel3` のような実装 module に
集中しています。

### よく使う kwargs / property を名前更新する

次は `v1.5.0` でも即 break にはなりませんが、このタイミングで置き換えるべきです。

| 旧 API / 旧引数 | 新 API / 新引数 |
| --- | --- |
| `shots=` | `n_shots=` |
| `interval=` | `shot_interval=` |
| `exp.linkup()` | `exp.connect()` |
| `exp.device_controller` | `exp.backend_controller` |
| `measurement.qubits` | `measurement.qubit_labels` |

例:

```python
# v1.4.8
result = exp.measure(sequence=sequence, shots=1024, interval=150 * 1024)

# v1.5.0
result = exp.measure(
    sequence=sequence,
    n_shots=1024,
    shot_interval=150 * 1024,
)
```

### contrib 系 helper を `Experiment` メソッドから切り離す

一部の specialized helper API は `Experiment` の直接メソッドではなくなり、
`qubex.contrib` に移動しました。旧メソッドは warning のあと
`NotImplementedError` を送出するため、直接の呼び出し箇所は必ず更新してください。

代表的な対応表:

| 旧 API | 新 API |
| --- | --- |
| `exp.rzx(...)` | `qx.contrib.rzx(exp, ...)` |
| `exp.rzx_gate_property(...)` | `qx.contrib.rzx_gate_property(exp, ...)` |
| `exp.measure_cr_crosstalk(...)` | `qx.contrib.measure_cr_crosstalk(exp, ...)` |
| `exp.cr_crosstalk_hamiltonian_tomography(...)` | `qx.contrib.cr_crosstalk_hamiltonian_tomography(exp, ...)` |
| `exp.measure_ghz_state(...)` | `qx.contrib.measure_ghz_state(exp, ...)` |
| `exp.measure_graph_state(...)` | `qx.contrib.measure_graph_state(exp, ...)` |
| `exp.measure_bell_states(...)` | `qx.contrib.measure_bell_states(exp, ...)` |
| `exp.purity_benchmarking(...)` | `qx.contrib.purity_benchmarking(exp, ...)` |
| `exp.interleaved_purity_benchmarking(...)` | `qx.contrib.interleaved_purity_benchmarking(exp, ...)` |
| `exp._stark_t1_experiment(...)` | `qx.contrib.stark_t1_experiment(exp, ...)` |
| `exp._stark_ramsey_experiment(...)` | `qx.contrib.stark_ramsey_experiment(exp, ...)` |
| `exp._simultaneous_measurement_coherence(...)` | `qx.contrib.simultaneous_coherence_measurement(exp, ...)` |

例:

```python
import qubex as qx

schedule = qx.contrib.rzx(
    exp,
    control_qubit="Q00",
    target_qubit="Q01",
    angle=0.78539816339,
)
```

### 可視化 import と結果アクセスを更新する

`v1.5.0` では result model に canonical な figure accessor が入りました。

次のように書き換えてください。

```python
# legacy payload access
fig = result["fig"]
figures = result["figures"]

# v1.5.0 canonical access
fig = result.figure
figures = result.figures
detail = result.get_figure("detail")
```

可視化 import も新 module に寄せてください。

```python
# legacy
from qubex.analysis import visualization as viz

# v1.5.0 canonical
import qubex.visualization as viz
```

model module の legacy import shim も残っていますが、新規 code では
`qubex.measurement.models` と `qubex.experiment.models` を使うのが正規です。

### 削除された内部 module への deep import をやめる

`qubex.pulse` や `qubex.simulator` の top-level export 自体は使えますが、
companion package 分離に伴って古い内部 module path の多くは消えています。

次のように更新してください。

```python
# v1.4.8 deep import
from qubex.pulse.library import Rect
from qubex.simulator.quantum_system import QuantumSystem

# v1.5.0 stable import
from qubex.pulse import Rect
from qubex.simulator import QuantumSystem
```

Qubex internals の上に再利用ライブラリを載せている場合は、削除されやすい
内部ファイル構成よりも `qxpulse`、`qxsimulator`、`qxcore`、
`qxvisualizer` などの companion package を直接参照する方が安全です。

## Timing と result model の更新

### 固定 `2 ns` 前提をやめる

`v1.5.0` では、主要な実行経路の timing を active backend から解決します。
hardcoded な `2` や `2.0` を sampling period として埋め込んでいる箇所は、
可能な限り backend 由来の値に置き換えてください。

推奨パターン:

```python
import numpy as np

wait_range = exp.util.discretize_time_range(
    np.geomspace(100, 100e3, 51),
    sampling_period=exp.measurement.sampling_period,
)
```

低レベル measurement result を扱う場合も、1 個の global 定数を前提にせず、
capture ごとの sampling metadata を使ってください。QuEL-3 向け script を
移植する場合は特に重要です。

### async / low-level flow では canonical measurement model を使う

同期の互換 API である `measure()` や `execute()` は、必要な箇所で引き続き
legacy の `MeasureResult` / `MultipleMeasureResult` を返します。一方、
async-first / low-level flow は `MeasurementResult`、`CaptureData`、
`SweepMeasurementResult` などの canonical model を返します。

これらは structured persistence に対応しています。

```python
result = await exp.run_measurement(schedule=schedule, n_shots=1024)
path = result.save("result.nc")
restored = type(result).load(path)
```

## 検証手順

移行後は、最小限でも次を実施してください。

1. Python `3.10+` の新しい環境を作り、必要な extra 付きで Qubex を入れる
2. `Experiment(system_id=..., config_dir=..., params_dir=...)` で実機 system を 1 つ読み込む
3. `exp.connect()` を実行し、必要なら `exp.configure()` も実行する
4. `measure()` または `execute()` で smoke test を 1 本流す
5. 以前 `2 ns` を前提にしていた sweep / notebook を 1 本動かす
6. moved helper API を使っているなら、contrib workflow も 1 本動かす
7. `chip_id`、`shots`、`interval`、legacy figure key、古い import path に関する warning が残っていないか確認する

## ロールバック時の注意

ロールバックが必要な場合は、次の順で戻してください。

1. `v1.4.8` 環境を復元する、または `v1.4.8` tag から再インストールする
2. file layout を変えた場合は、以前の設定スナップショットへ戻す
3. `qubex.system`、`qubex.contrib`、backend 由来 timing に依存した notebook /
   script の変更を戻す

`v1.5.0` では legacy input がいくつか互換用に残っているため、段階的移行も可能です。
まず import と runtime selector を更新し、その後 parameter file と
warning が出る call site を順に移していく方法でも進められます。
