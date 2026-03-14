# システム設定

1 つの具体的な装置構成を選ぶ識別子として、`system_id` を使ってください。
1 つの system には、チップのメタデータ、バックエンド種別、バックエンド固有の実行設定、そして mux と物理ポートの対応をまとめます。

`chip_id` と `system_id` はユーザー定義のラベルです。Qubex はどちらにも特定の命名規則を要求しません。

設定ファイルを自分で管理する場合は、`config_dir` と `params_dir` の両方を明示的に渡してください。これにより、ファイル配置を自分で制御でき、古いパス規約への依存も避けられます。

これらを省略した場合、Qubex は次の順序で config root を解決します。

1. `QUBEX_CONFIG_ROOT`
2. `~/qubex-config`
3. `/opt/qubex-config`
4. 旧来の `/home/shared/qubex-config`

どれも存在しない場合、Qubex は `~/qubex-config` を既定値として使います。

## 推奨ディレクトリ構成

共有カタログは 1 つの config ディレクトリにまとめ、パラメータファイルは実行したい system ごとにディレクトリを分ける構成を推奨します。

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
    skew.yaml  # QuEL-1/QuBE 向け
  params/
    SYSTEM_A/
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      capture_delay.yaml
      ...
  calibration/
    SYSTEM_A/
      calib_note.json
```

- `config/` には共有の system catalog を置きます。
- `params/<system_id>/` の各ファイルは、1 つのパラメータファミリを表します。
- `calibration/<system_id>/calib_note.json` は既定の calibration note の保存先です。
- `skew.yaml` は任意ですが、QuEL-1 の skew calibration ワークフローでは通常必要になります。

## 共有 config ファイルを定義する

### `chip.yaml`

チップのメタデータを chip ごとに 1 回だけ定義します。

```yaml
CHIP_A:
  name: "Example chip"
  n_qubits: 64
  topology:
    type: square_lattice
    mux_size: 4
```

### `box.yaml`

wiring に登場しうるハードウェアユニットを登録します。

```yaml
BOX_A:
  name: "quel3-02-a01"
  type: "quel3"

BOX_B:
  name: "QuEL-1 #5-01"
  type: "quel1-a"
  address: "10.1.0.73"
  adapter: "500202A50TAAA"

BOX_C:
  name: "QuEL-1 SE R8 #1"
  type: "quel1se-riken8"
  address: "10.1.0.160"
  adapter: "500202A800RAA"
  options:
    - "se8_mxfe1_awg2222"
```

QuEL-3 のエントリでは `address` と `adapter` は任意です。QuBE と QuEL-1 のエントリでは必須です。

`options` は任意で、box に対するバックエンドオプションラベルのリストを受け取ります。非既定のハードウェアプロファイルが必要なときに使ってください。

例えば `quel1se-riken8` は `se8_mxfe1_awg1331`、`se8_mxfe1_awg2222`、`se8_mxfe1_awg3113` のような AWG プロファイルラベルを受け取れます。AWG プロファイルが指定されない場合、Qubex は `se8_mxfe1_awg2222` を使います。

### `system.yaml`

実行可能な構成ごとに 1 エントリを作成します。複数の system が同じ `chip_id` を参照しても構いません。

```yaml
SYSTEM_A:
  chip_id: CHIP_A
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
```

- トップレベルのキーが `system_id` です。
- `backend` で、この system が使うバックエンド種別を選びます。
- バックエンド固有セクションの名前は `backend` と同じにします。

### `wiring.yaml`

`system_id` をキーにし、mux ごとに 1 行ずつ wiring を定義します。

```yaml
SYSTEM_A:
  - mux: 0
    ctrl: [BOX_A:4, BOX_A:2, BOX_A:11, BOX_A:9]
    read_out: BOX_A:1
    read_in: BOX_A:0
  - mux: 1
    ctrl: [BOX_A:16, BOX_A:14, BOX_A:17, BOX_A:15]
    read_out: BOX_A:8
    read_in: BOX_A:7
```

Qubex は `wiring.yaml` で `BOX:PORT` と `BOX-PORT` の両方を受け付けますが、保守しやすさのためにはどちらかに統一するのがよいです。

## パラメータファイルを定義する

現在の形式では、パラメータファミリごとに 1 つの構造化 YAML ファイルを使います。各ファイルは、注釈用の `meta` と実データ用の `data` を持ちます。

```yaml
meta:
  description: Example low-frequency control frequencies
  unit: GHz
data:
  0: 3.000
  1: 3.031
  2: 3.062
```

- ファイルは、Qubex に渡す `params_dir` の下に置いてください。
- qubit 単位のパラメータでは、キーに `0` や `1` のような整数インデックス、あるいは `Q000` や `Q001` のようなラベルを使えます。
- `meta.unit` を設定すると、Qubex は値を内部の基底単位へ変換します。周波数系は `GHz`、時間系は `ns` です。
- `meta.default` を設定すると、`data` 中の `None` はその既定値にフォールバックします。

旧来の `params.yaml` と `props.yaml` も互換入力としてサポートされています。旧形式の map と分割 YAML が両方ある場合、Qubex はまず分割 YAML を読み込み、足りないキーだけ旧形式ファイルから補います。

## コードから設定を読み込む

具体的な `system_id`、共有 config ディレクトリ、選択した parameter ディレクトリを渡してください。

```python
import qubex as qx

exp = qx.Experiment(
    system_id="SYSTEM_A",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)
```

同じファイル群は `ConfigLoader` を使って直接読み込んで確認することもできます。

```python
from qubex.system import ConfigLoader

loader = ConfigLoader(
    system_id="SYSTEM_A",
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)

system = loader.get_experiment_system()
```

ファイルを用意したら、次は [クイックスタート](quickstart.md) に進んでください。
