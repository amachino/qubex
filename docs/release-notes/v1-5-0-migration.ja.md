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
- simulator の `Control` 補間に依存している、または control の segment data を
  in-place で変更している

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
- simulator の `Control` 補間を、明示的に sample した waveform へ置き換える
- sweep、plot、timing utility で固定 `2 ns` を使わないようにする

## インストールと実行環境の変更

`v1.5.0` の repository workflow は `uv` 管理環境を前提にしています。
具体的な導入手順は
[インストールガイド](../user-guide/getting-started/installation.md) を参照してください。

最低限、次の前提は更新してください。

- Python `3.9` は非対応になりました。Python `3.10` 以上を使ってください。
- 実機向け依存関係は `backend` extra で導入します。
- repository 内開発は `uv` 環境で `make sync` を前提にしています。
- `qxsimulator` は JAX、Optax、IPython を install しなくなりました。JAX と Optax は
  deprecated となった `PulseOptimizer` だけが使用していました。この API を移行期間中も
  使用する場合は、この 2 package を別途 install してください。IPython の display 連携は
  削除されました。

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
      measurement_defaults.yaml
```

`params.yaml` と `props.yaml` は `v1.5.0` でも fallback input として読まれます。
一度に全部移行する必要はありませんが、新規更新分は per-file 形式へ寄せるのが
推奨です。

system ごとに `n_shots`、`shot_interval`、readout timing の既定値を変えたい場合は、
`params/<system_id>/measurement_defaults.yaml` を使ってください。

### `configuration_mode` と control port の channel 数を見直す

`configuration_mode` は channel role の優先順として解釈されるようになりました。

- `ge-ef-cr` は `ge`、`ef`、`cr`
- `ge-ef-fh` は `ge`、`ef`、`fh`
- `ge-cr-cr` は `ge`、`cr`、`cr`
- control port の channel 数が足りない場合は左側の役割だけを残します

そのため、ハードウェアプロファイルが control port の channel 数を変えると、
実際に生成される target も変わります。例えば QuEL-1 SE R8 の
`se8_mxfe1_awg2222` は profile-controlled port を `2-2-2-2` にするため、
`configuration_mode="ge-ef-cr"` はそこで `ge-ef` target を生成します。
その port で CR target が必要なら、
`configuration_mode="ge-cr-cr"` を使ってください。2 channel port で EF/FH workflow
を使う場合は `configuration_mode="ge-ef-fh"` を指定します。EF と FH は 2 本目の
channel を共有します。

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

### simulator の `Control` sampling を更新する

simulator の `Control` は、有限時間の区分定数信号を表すようになりました。
constructor の `interpolation` 引数と `interpolator` property は削除されています。
zero-order hold された信号の評価には `get_samples()` を使ってください。

```python
# v1.4.x
control = Control(..., interpolation="linear")
samples = control.interpolator(times)

# v1.5.0
control = Control(...)
samples = control.get_samples(times)
```

内部 segment 境界では、`get_samples()` はその境界から始まる segment を返します。
control の開始前と全 duration を越えた時刻ではゼロを返します。linear、cubic、
FIR 相当の再構成に依存していた場合は、`Control` を作る前に必要な細かさの waveform
を生成し、対応する segment duration とともに渡してください。

`Control` は `waveform` と `durations` を copy し、read-only array として公開します。
これらを in-place で変更せず、新しい `Control` を作ってください。各 segment の
duration は有限かつゼロより大きい必要があります。空の control では、空の waveform
と duration array を引き続き使用できます。

### `simulate()` の伝播設定を更新する

`QuantumSimulator.simulate()` は、`dt` を一様な出力間隔の保証ではなく、最大伝播幅
として解釈するようになりました。積分グリッドは、一様な `dt` グリッドとすべての
`Control` segment 境界、および要求したすべての出力時刻を組み合わせます。そのため、
境界や出力時刻によって `dt` より短い区間が追加されることがあります。

`TIME_STEP` 定数を削除しました。`simulate()` は default を `dt=0.1` として直接
宣言します。異なる最大伝播幅が必要な場合は、`dt` を明示してください。

各区間では、zero-order hold された control 振幅を左端点で選択します。連続的に
時間依存する carrier 項と coupling 項は区間の中点で評価します。このため、離調した
drive や回転する coupling の結果は、従来の左端点による伝播から変わる場合があります。

`Control.frame_shifts` と `Control.final_frame_shift` は logical frame の metadata として
保持され、states や propagators へ物理的な回転として適用されません。`PulseSchedule`
の途中の frame shift は、後続 waveform sample の位相へすでに反映されています。segment
ごとの metadata は、それとは別に `SimulationResult` が返却 trajectory を変化する
logical frame で解釈するために使います。`n_samples` を指定する場合は、物理的な時間発展
の初期点と終端点の両方を保持するため、2 以上にしてください。正の control duration
では、result は時刻ゼロから共通 duration までを等間隔に分けた、ちょうど
`n_samples` 個の時刻を含みます。duration がゼロなら初期点だけを含みます。
`n_samples` を省略した場合は、固定 step 積分のすべての点を返します。

### QuTiP solver の積分設定を `options` で指定する

QuTiP ベースの `QuantumSimulator.sesolve()`、`mesolve()`、`propagator()`、
`gate_fidelity()`、`create_simulation_parameters()`、
`create_simulation_model()` の signature から `dt` 引数を削除しました。`dt` を
渡す既存の呼び出しは互換性のために受理され、`DeprecationWarning` を発行して
値を無視します。model の時刻列は、すべての `Control` segment 境界の和集合に
なりました。control 振幅は境界間で厳密な zero-order hold とし、drive frame と
coupling の連続位相は QuTiP の解析的な coefficient として保持します。この時刻列は
`SimulationModel.boundary_times` として公開され、
`create_simulation_parameters()` の返却 dict では `boundary_times` entry になります。
従来の汎用的な `times` 名は使用しません。

`sesolve()` と `mesolve()` で `n_samples` を指定すると、正の control duration では
等間隔な公開出力時刻をちょうどその個数だけ要求します。Qubex は、それらの出力時刻と
すべての control 境界の和集合を QuTiP に渡し、要求した出力 trajectory だけを result
に残します。したがって、zero-order hold の各不連続点を solver checkpoint として
守りながら、公開 result を非一様な control grid に固定せずに済みます。duration が
ゼロなら初期点だけを含みます。`n_samples` を省略した場合は、従来どおりすべての
control 境界を返します。

内部の積分幅は QuTiP が適応的に決定します。`method`、`rtol`、`atol`、
`max_step` などの solver 設定は `options` で指定してください。`max_step` を省略
した場合は、最短 control segment duration の半分を既定値とし、明示した値を
優先します。`nsteps` を省略した場合は、2500 と、最長 solver interval を
`max_step` で進むために最低限必要な step 数の 2 倍のうち、大きい方を使用します。
それ以外の積分法や誤差許容値には QuTiP の既定値を使用します。`dt` が意味を持つ
のは `QuantumSimulator.simulate()` だけです。

`QuantumSimulator.propagator()` は、全 `Control` segment 境界の和集合における
累積 propagator の list を返します。全時間発展だけが必要な場合は、list の最終要素を
使用してください。各境界まで順に積分するため、区分定数 control の各不連続点も
solver interval の境界になります。閉鎖系では list の要素は Hilbert 空間で計算した
unitary operator であり、正の decoherence rate を1つでも持つ開放系では Liouville
空間で計算した superoperator です。rate がゼロの relaxation operator と dephasing
operator は model に追加しません。fidelity method は最終 propagator を使用し、
どちらの表現も受け取ります。既定では computational subspace の map を切り出し、
`levels="full"` では物理空間全体を、object ごとの level mapping では qudit や
非 computational subspace を評価します。これにより、閉鎖系では大きな
Liouville-space 積分を回避します。

`gate_fidelity()` は deprecated です。代わりに `average_gate_fidelity()` を使用して
ください。deprecated 名は互換期間中、同じ計算を行う alias として残ります。
`process_fidelity()` は、切り出した computational-subspace map と target unitary の
normalized Choi overlap を返します。subspace の切り出し後は map が trace-decreasing
になり得るため、`average_gate_fidelity()` は leakage を失敗として数え、
$F_\mathrm{avg}=(dF_\mathrm{pro}+p_\mathrm{surv})/(d+1)$ を使用します。ここで
$p_\mathrm{surv}=\operatorname{Tr}[\mathcal{E}_\mathrm{sub}(I)]/d$ です。
trace-preserving map では $p_\mathrm{surv}=1$ となり、QuTiP の標準的な
average-gate-fidelity の関係式に一致します。

`QuantumSystem.unitary()` を使うと、object label で target を指定し、物理 Hilbert
空間全体へ embed できます。

```python
from qxsimulator import gates

target = system.unitary({"Q04-Q01": "CZ"})
fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

文字列では既存の Qubex Clifford gate 名と一般的な static gate を指定できます。
引数を持つ gate は `gates.rotation(generator, angle)` で作ります。この関数は
`exp(-1j * angle * generator / 2)` を計算します。`X`、`Y`、`Z`、`XX`、`YY`、
`ZZ`、`ZX` の生成子を直接組み合わせられ、例えば
`gates.rotation((gates.XX + gates.YY) / 2, angle)` と書けます。fidelity method には
object label の mapping を直接渡せます。`Qobj` target は、選択した subspace と
同じ次元でも、物理 system 全体の次元でも指定できます。

`PulseOptimizer` は deprecated であり、将来の release で削除予定です。JAX と Optax は
`qxsimulator` と一緒には install されなくなったため、この互換 API を使う場合は別途
install してください。IPython の display 連携は削除され、通常の simulator import と
workflow ではこれらの package を読み込みません。

### propagator trajectory を明示的に要求する

`SimulationResult.states` と `SimulationResult.propagators` は、QuTiP の
`Qobj` instance の list になりました。`SimulationResult.unitaries` は deprecated
です。代わりに `propagators` を使用してください。deprecated attribute は互換期間中、
同じ list を返す alias として残ります。

`SimulationResult.control_frequencies` も deprecated です。1つの target に異なる
周波数の Control が複数存在し得るため、代わりに `SimulationResult.controls` を直接
確認してください。`frame="drive"` を指定した場合、対象に distinct な Control 周波数が
ちょうど1つ存在するときだけ解析 frame を自動推定します。Control がない場合や
multi-tone の場合は、`frame_frequency` を GHz で明示してください。

`SimulationResult.get_substates()` は、object dtype の NumPy array ではなく、文書化
された result model に合わせて `list[Qobj]` を返します。Bloch vector と density
matrix の helper は、引き続きそれぞれ `float64` と `complex128` の数値 NumPy array
を返します。
substate 抽出 method の `frame`、`frame_frequency`、`apply_frame_shifts` は keyword-only
です。positional に渡している呼び出しは、argument 名を明示する形へ更新してください。

`SimulationResult` は構築時に trajectory の対応関係と system dimensions を検証します。
渡された Control、state、propagator の container は copy し、times は copy した
read-only の `float64` array として保持します。times は finite かつ狭義単調増加である
必要があり、不正な result object は構築時に `ValueError` になります。等値比較は
identity based とし、`repr()` は大きな array や QuTiP object を展開せず trajectory の
件数を表示します。

`QuantumSimulator.simulate()` は default で propagator を計算します。state
trajectory だけを保持する場合は、`compute_propagators=False` を指定してください。
`QuantumSimulator.sesolve()` と `mesolve()` は default では propagator を計算しません。
両方の trajectory が必要な場合に明示的に要求します。

```python
result = simulator.sesolve(
    controls,
    compute_propagators=True,
)
```

`sesolve()` の各 propagator は ket に作用する operator です。`mesolve()` の各
propagator は vector 化した density matrix に作用する superoperator です。完全な
propagator の計算は、1つの state の時間発展より高コストです。特に `mesolve()` の
superoperator は Hilbert 空間の次元を `d` とすると `d ** 4` 要素を持ちます。
`propagators` が空の list の場合、その trajectory は計算されていません。

states と propagators は simulator の物理的な rotating frame に保持されます。
`PulseSchedule` から変換した Control は、segment ごとの `frame_shifts` と終端の
`final_frame_shift` を座標系 metadata として保持します。
`SimulationResult.get_substates()` および density matrix・Bloch vector の helper は、
default で各返却時刻の累積 frame shift を適用します。物理 frame の raw trajectory を
確認するには `apply_frame_shifts=False` を指定してください。内部境界では、その境界から
始まる segment の shift を使い、最終境界以降では終端 shift を使います。

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
