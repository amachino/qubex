# パラメータファイル

`params/<system_id>/` ディレクトリには、1 つの実行可能な system に固有の
パラメータを置きます。`chip.yaml`、`system.yaml`、`wiring.yaml`、
`skew.yaml` などの共有設定ファイルとは分けて管理します。

Qubex は `Experiment` または `ConfigLoader` が system を load するときに
parameter file を読み込みます。その後で YAML を編集した場合は、新しい
`Experiment` を作るか、`ConfigLoader.load()` を再実行してください。

## ディレクトリ構成

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
    skew.yaml
  params/
    SYSTEM_A/
      qubit_frequency.yaml
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      measurement_defaults.yaml
      capture_delay.yaml
      ...
```

選択した system のディレクトリを `params_dir` として渡してください。

```python
import qubex as qx

exp = qx.Experiment(
    system_id="SYSTEM_A",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)
```

`params/<system_id>/` の下に Qubex が認識しないファイルを置いても、ユーザー
コードが明示的に開かない限り Qubex はそれを読みません。

## 構造化パラメータ形式

ほとんどの parameter file は `meta` / `data` 形式を使います。

```yaml
meta:
  description: Example GE control frequencies
  unit: GHz
  default: null
data:
  0: 4.912
  1: 4.846
  2: null
```

- `meta` は注釈と loader option のための領域です。
- `data` には Qubex が読み込む実データを書きます。
- `meta.unit` を指定すると、top-level の数値が Qubex の内部単位へ変換されます。
- `meta.default` を指定すると、`data` 中の `null` と YAML NaN が unit 変換前に
  その値へ置き換えられます。

認識される unit 変換は次の通りです。

| 種類 | 受け付ける unit | 内部単位 |
| --- | --- | --- |
| 周波数 | `Hz`, `kHz`, `MHz`, `GHz` | `GHz` |
| 時間 | `s`, `ms`, `us`, `ns` | `ns` |

未認識の unit は scale `1.0` として扱われます。ただし `capture_delay` 系の
ファイルは backend ごとの unit validation を受けます。`meta.unit` による変換は
top-level の数値だけに適用されるため、`jpa_params.yaml` のような nested map は
下で説明する内部単位で直接書いてください。

### キー

qubit 単位のファイルでは、整数 index、数字文字列、または具体的な qubit label を
キーにできます。

```yaml
data:
  0: 4.912      # chip の qubit label に正規化される
  "1": 4.846    # これも正規化される
  Q002: 4.901   # 文字列 label はそのまま使われる
```

数字キーの場合、Qubex は `chip.yaml:n_qubits` から label 幅を決めます。
例えば 64 qubit chip は `Q00` のような label、144 qubit chip は `Q000` のような
label を使います。文字列 label はそのまま受け付けられるため、選択した chip の
label と一致している必要があります。

mux 単位のファイルでは mux index を使います。pair 単位のファイルでは、
`Q000-Q001` のように、その値を読む workflow が期待する pair label を使います。

## 読み込み優先度

現在の推奨形式は、parameter family ごとに 1 つの YAML ファイルを置く形式です。
旧形式の `params.yaml` と `props.yaml` も互換入力として引き続きサポートされます。

認識される各 parameter について、読み込み順は次の通りです。

1. `<parameter>.yaml` が存在し、`data` が空でなければ、それを読み込みます。
2. per-file YAML に無いキーだけ、`params.yaml` または `props.yaml` の legacy entry
   で補います。
3. per-file YAML が存在しても `data` が空の場合、その parameter については legacy
   map へフォールバックします。
4. per-file YAML が存在しない場合、legacy map だけを使います。

この merge は浅い key 単位の merge です。同じキーがある場合、per-file YAML が
legacy entry より優先されます。

## 周波数の優先度

物理的に関連していても、同じ意味ではない file があります。Qubex は target を作る
とき、次の優先度を使います。

| 実行時に使われる値 | 優先 source | fallback source | 主な用途 |
| --- | --- | --- | --- |
| Qubit GE 周波数 | `control_frequency.yaml` | `qubit_frequency.yaml` | GE target と、target qubit がある CR target |
| Qubit bare 周波数 | `qubit_frequency.yaml` | なし | `Qubit.bare_frequency` と GE fallback |
| Qubit EF 周波数 | `control_frequency_ef.yaml` | 有効な GE 周波数 + anharmonicity | EF target |
| Resonator readout 周波数 | `readout_frequency.yaml` | `resonator_frequency.yaml` | readout generator target と capture target |
| Resonator ground-state 周波数 | `resonator_frequency.yaml` | なし | `Resonator.frequency_g` と readout fallback |

したがって、同じ qubit に対して `control_frequency.yaml` に有限値がある間は、
`qubit_frequency.yaml` を編集しても GE target 周波数は変わりません。同様に、
同じ resonator に対して `readout_frequency.yaml` に有限値がある間は、
`resonator_frequency.yaml` を編集しても readout target 周波数は変わりません。

## 認識されるファイル

以下は `ConfigLoader` が認識する parameter file です。

### 量子系と周波数パラメータ

| ファイル | scope | unit | 効果 | legacy source |
| --- | --- | --- | --- | --- |
| `qubit_frequency.yaml` | qubit | 周波数、通常 `GHz` | bare qubit 周波数と GE fallback。 | `props.yaml:<chip_id>.qubit_frequency` |
| `qubit_anharmonicity.yaml` | qubit | 周波数、通常 `GHz` | qubit anharmonicity。無い場合、model は有効な GE 周波数から fallback 値を導出します。 | `props.yaml:<chip_id>.anharmonicity` |
| `control_frequency.yaml` | qubit | 周波数、通常 `GHz` | 優先される GE control 周波数。GE target と target qubit がある CR target に使われます。 | `props.yaml:<chip_id>.control_frequency` |
| `control_frequency_ef.yaml` | qubit | 周波数、通常 `GHz` | 優先される EF control 周波数。無い場合、GE 周波数 + anharmonicity を使います。 | `props.yaml:<chip_id>.control_frequency_ef` |
| `resonator_frequency.yaml` | qubit または qubit-keyed resonator | 周波数、通常 `GHz` | resonator ground-state 周波数と readout fallback。キーは qubit label または index です。 | `props.yaml:<chip_id>.resonator_frequency` |
| `readout_frequency.yaml` | qubit または qubit-keyed resonator | 周波数、通常 `GHz` | 優先される readout drive / capture 周波数。キーは qubit label または index です。 | `props.yaml:<chip_id>.readout_frequency` |

例:

```yaml
meta:
  description: Tuned GE control frequencies
  unit: GHz
data:
  0: 4.9123
  1: 4.8461
```

### 制御・ハードウェアパラメータ

| ファイル | scope | unit | 効果 | legacy source |
| --- | --- | --- | --- | --- |
| `frequency_margin.yaml` | target type | 周波数、通常 `GHz` | target deployment 時の周波数 margin を target type ごとに上書きします。key は `READ`, `CTRL_GE`, `CTRL_EF`, `CTRL_CR`, `PUMP` です。 | `params.yaml:<chip_id>.frequency_margin` |
| `control_amplitude.yaml` | qubit | 無次元 | 既定の control pulse amplitude。 | `params.yaml:<chip_id>.control_amplitude` |
| `readout_amplitude.yaml` | qubit | 無次元 | 既定の readout pulse amplitude。 | `params.yaml:<chip_id>.readout_amplitude` |
| `control_vatt.yaml` | qubit | 整数または `null` | 対応 backend の control-line VATT 設定。 | `params.yaml:<chip_id>.control_vatt` |
| `readout_vatt.yaml` | mux | 整数または `null` | 対応 backend の readout-line VATT 設定。 | `params.yaml:<chip_id>.readout_vatt` |
| `pump_vatt.yaml` | mux | 整数または `null` | 対応 backend の pump-line VATT 設定。 | `params.yaml:<chip_id>.pump_vatt` |
| `control_fsc.yaml` | qubit | 整数または `null` | 対応 backend の control-line full-scale current 設定。 | `params.yaml:<chip_id>.control_fsc` |
| `readout_fsc.yaml` | mux | 整数または `null` | 対応 backend の readout-line full-scale current 設定。 | `params.yaml:<chip_id>.readout_fsc` |
| `pump_fsc.yaml` | mux | 整数または `null` | 対応 backend の pump-line full-scale current 設定。 | `params.yaml:<chip_id>.pump_fsc` |
| `capture_delay.yaml` | mux | backend 依存 | capture timing offset。QuEL-1 では `meta.unit: ndelay`、QuEL-3 では `meta.unit: ns` が必須です。 | `params.yaml:<chip_id>.capture_delay` |
| `capture_delay_word.yaml` | mux | `word` または `words` | QuEL-1 の capture-delay word offset。QuEL-3 ではサポートされません。 | `params.yaml:<chip_id>.capture_delay_word` |
| `jpa_params.yaml` | mux | 内部単位 | JPA parameter。各 mux entry は `pump_frequency` (`GHz`), `pump_amplitude`, `dc_voltage` を持てます。無い field は backend default で補われます。 | `params.yaml:<chip_id>.jpa_params` |

QuEL-1 では VATT、FSC、capture delay、JPA fields に対して QuEL-1 default が
materialize されます。QuEL-3 では未対応の VATT/FSC/capture-delay-word は `null`
として materialize され、frequency margin と JPA fields には QuEL-3 default が
使われます。

例:

```yaml
# QuEL-3 用 capture_delay.yaml
meta:
  description: Capture timing offsets
  unit: ns
data:
  0: 8.0
  1: 8.0
```

```yaml
# QuEL-1 用 capture_delay.yaml
meta:
  description: Capture timing offsets
  unit: ndelay
data:
  0: 7
  1: 8
```

```yaml
# jpa_params.yaml
meta:
  description: JPA pump and bias settings
data:
  0:
    pump_frequency: 6.000
    pump_amplitude: 0.10
    dc_voltage: 0.00
```

### Measurement defaults

`measurement_defaults.yaml` は `meta` / `data` 形式ではありません。system ごとの
measurement 実行条件と readout timing に対する partial default file です。

```yaml
schema_version: 1

execution:
  n_shots: 2048
  shot_interval_ns: 200000.0

readout:
  duration_ns: 512.0
  ramp_time_ns: 24.0
  pre_margin_ns: 16.0
  post_margin_ns: 96.0
```

| field | 意味 |
| --- | --- |
| `schema_version` | 指定する場合は `1` である必要があります。 |
| `execution.n_shots` | configured default を使う measurement API で明示引数が無い場合の shot 数。正の値が必要です。 |
| `execution.shot_interval_ns` | configured default を使う measurement API で明示引数が無い場合の shot interval (`ns`)。正の値が必要です。 |
| `readout.duration_ns` | 既定の readout duration (`ns`)。非負値が必要です。 |
| `readout.ramp_time_ns` | 既定の readout ramp time (`ns`)。非負値が必要です。 |
| `readout.pre_margin_ns` | 既定の pre-readout margin (`ns`)。非負値が必要です。 |
| `readout.post_margin_ns` | 既定の post-readout margin (`ns`)。非負値が必要です。 |

API に明示的に渡した引数は `measurement_defaults.yaml` より優先されます。一部の
legacy または特殊な measurement path では、lower-level の measurement config に
届く前に関数内の local default が設定される場合があります。default が反映されない
場合は、その関数の signature と呼び出し経路を確認してください。

### Characterization / diagnostic properties

次のファイルは `ConfigLoader.load_param_data(...)` で認識されます。base system
loader はこれらを target frequency や hardware setting には変換しませんが、
characterization や workflow code が system property として使えます。

| ファイル | scope | unit | 意味 | legacy source |
| --- | --- | --- | --- | --- |
| `t1.yaml` | qubit | 時間、通常 `ns` または `us` | T1 推定値。 | `props.yaml:<chip_id>.t1` |
| `t2_echo.yaml` | qubit | 時間、通常 `ns` または `us` | Echo T2 推定値。 | `props.yaml:<chip_id>.t2_echo` |
| `t2_star.yaml` | qubit | 時間、通常 `ns` または `us` | Ramsey T2* 推定値。 | `props.yaml:<chip_id>.t2_star` |
| `t2_star_ef.yaml` | qubit | 時間、通常 `ns` または `us` | EF Ramsey T2* 推定値。 | `props.yaml:<chip_id>.t2_star_ef` |
| `average_readout_fidelity.yaml` | qubit | 無次元 | average readout fidelity 推定値。 | `props.yaml:<chip_id>.average_readout_fidelity` |
| `quantum_efficiency.yaml` | qubit | 無次元 | quantum efficiency 推定値。 | `props.yaml:<chip_id>.quantum_efficiency` |
| `x90_gate_fidelity.yaml` | qubit | 無次元 | X90 gate fidelity 推定値。 | `props.yaml:<chip_id>.x90_gate_fidelity` |
| `x180_gate_fidelity.yaml` | qubit | 無次元 | X180 gate fidelity 推定値。 | `props.yaml:<chip_id>.x180_gate_fidelity` |
| `zx90_gate_fidelity.yaml` | pair | 無次元 | ZX90 gate fidelity 推定値。 | `props.yaml:<chip_id>.zx90_gate_fidelity` |
| `static_zz_interaction.yaml` | pair | 周波数、通常 `GHz` または `MHz` | static ZZ interaction 推定値。 | `props.yaml:<chip_id>.static_zz_interaction` |
| `qubit_qubit_coupling_strength.yaml` | pair | 周波数、通常 `GHz` または `MHz` | qubit-qubit coupling 推定値。 | `props.yaml:<chip_id>.qubit_qubit_coupling_strength` |
| `resonator_external_linewidth.yaml` | qubit または qubit-keyed resonator | 周波数、通常 `GHz` または `MHz` | resonator external linewidth 推定値。 | `props.yaml:<chip_id>.external_loss_rate` |
| `resonator_internal_linewidth.yaml` | qubit または qubit-keyed resonator | 周波数、通常 `GHz` または `MHz` | resonator internal linewidth 推定値。 | `props.yaml:<chip_id>.internal_loss_rate` |

## `params/` ではないファイル

共有 system file は `params/<system_id>/` には置かないでください。

- `skew.yaml` は共有 `config/` ディレクトリのファイルです。QuEL-1 / QuBE の
  box 間 timing を制御し、waveform timing に影響し得ますが、parameter family
  file ではありません。
- `calibration/<system_id>/calib_note.json` は既定の calibration note 保存先です。
  このページで説明している parameter file とは別です。

## よくある落とし穴

- `params.yaml` や `props.yaml` が per-file YAML を上書きすると期待すること。
  同じキーでは per-file YAML が優先されます。
- `control_frequency.yaml` が同じ qubit を定義している状態で
  `qubit_frequency.yaml` だけを編集すること。GE と CR target は
  `control_frequency.yaml` を先に使います。
- `readout_frequency.yaml` が同じ resonator を定義している状態で
  `resonator_frequency.yaml` だけを編集すること。readout target は
  `readout_frequency.yaml` を先に使います。
- `capture_delay.yaml` の `meta.unit` を省略したり、QuEL-3 system に QuEL-1 用の
  unit を使ったりすること。
- `Experiment` を作った後に YAML を編集し、既存 object が自動更新されると期待すること。
