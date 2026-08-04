# システム設定

1 つの具体的な装置構成を選ぶ識別子として、`system_id` を使ってください。
1 つの system には、チップのメタデータ、バックエンド種別、バックエンド固有の実行設定、そして読み出し MUX と物理ポートの対応をまとめます。

`chip_id` と `system_id` はユーザー定義のラベルです。Qubex はどちらにも特定の命名規則を要求しません。

設定ファイルを自分で管理する場合は、`config_dir` と `params_dir` の両方を明示的に渡してください。これにより、ファイル配置を自分で制御でき、古いパス規約への依存も避けられます。

これらを省略した場合、Qubex は次の順序で config root を解決します。

1. `QUBEX_CONFIG_ROOT`
2. `~/qubex-config`
3. `/opt/qubex-config`
4. 旧来の `/home/shared/qubex-config`

どれも存在しない場合、Qubex は `~/qubex-config` を既定値として使います。

## 推奨ディレクトリ構成

共有カタログは 1 つの config ディレクトリにまとめ、システム固有のパラメータファイルは実行したいシステムごとにディレクトリを分ける構成を推奨します。

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
    external_devices.yaml  # 外部計測器を使う場合
    skew.yaml  # QuEL-1/QuBE 向け
  params/
    SYSTEM_A/
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      measurement_defaults.yaml
      capture_delay.yaml
      ...
  calibration/
    SYSTEM_A/
      calib_note.json
```

- `config/` には共有のシステム設定ファイルを置きます。
- `params/<system_id>/` の各ファイルは、1 つのパラメータファミリを表します。
- `calibration/<system_id>/calib_note.json` は既定の較正ファイルの保存先です。
- `external_devices.yaml` は任意で、JPA バイアス用 DC 電圧源など、QuEL 制御系の外にある計測器を設定します。
- `skew.yaml` は任意ですが、複数の QuEL-1 制御装置を用いた同期実験を行う場合に必要になります。

## 共有設定ファイルを定義する

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

### `external_devices.yaml`

このファイルはシステム設定本体と同じ構成です: `devices` が `box.yaml` と同じ発想でデバイスを定義し、`wiring` が `wiring.yaml` と同じ発想で mux とデバイス出力を接続し、`settings` が制御ポリシーを持ちます。

```yaml
devices:
  ONS1:
    driver: ons61797
    channels: [1, 2]
    params:
      port: /dev/ttyACM0

wiring:
  - mux: 6
    bias: ONS1-1
  - mux: 7
    bias: ONS1-2

settings:
  ramp:
    rate_v_per_s: 0.1
    step_interval_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  idle_voltage_v: 0.0
  overrides:
    - mux: 7
      ramp:
        rate_v_per_s: 0.05
```

- `devices` — `driver` が adapter を選び、`channels` がデバイスの出力を列挙します。`params` の中身は driver 固有で、選択した driver 自身が検証します (ONS61797: serial は `port`、network は `ip_address`。両方は不可)。
- `wiring` — 各エントリは役割名 (`bias`) と `デバイス名-チャンネル` 形式の出力参照 (`ONS1-2` = デバイス `ONS1` の channel 2、1 始まり) を持ちます。操作できるのは配線済みの出力だけで、未配線の mux や `channels` 外のチャンネルは推測せず明示エラーになります。
- `settings` — 直下の値が配線済み全 mux の既定値、`overrides` が mux 単位の上書きです。`idle_voltage_v` は未使用時にこの線が待機する値です。1 つの `settings` (役割は既定 `bias`、`role` で変更可) が拾う出力はすべて同じデバイスにある必要があります。

context 終了時の挙動は設定ではなく呼び出し側で選びます: `measurement.apply_dc_voltages(..., on_exit=...)` に `"idle"` (既定 — アイドル電圧まで ramp) か `"hold"` (バイアスを残す) を渡します。出力スイッチが自動で切り替わることはありません — 変えるのは明示的な `turn_on()` / `turn_off()` だけです。

Qblox SPI Rackを、USB接続を所有するserver processを経由して制御する場合は、
`qblox_server` driverを使用します。QubexはこのserverへTCP clientとして
接続します。serial deviceを開くprocessを一つに保てるため、複数のシステムが
同じ装置を利用する環境ではこの構成を推奨します。

```yaml
devices:
  Qblox1:
    driver: qblox_server
    channels: [1, 2]
    params:
      host: "<qblox-backend-host>"
      port: <qblox-backend-port>
      timeout_s: 1200

wiring:
  - mux: 6
    bias: Qblox1-1

settings:
  ramp:
    rate_v_per_s: 0.1
    step_interval_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  idle_voltage_v: 0.0
```

serverは各出力を `<デバイス名>-<channel>` で識別するため、デバイス名はbackendの報告名に合わせます (`Qblox1` → `Qblox1-15`)。命名が規則的でない場合は `params` の `device_names: {channel: 名前}` で対応します。rampはserver側で一括実行されるため他clientは割り込めませんが、完了まで他channelの操作が待つことがあります。認証のないsocketを信頼できないnetworkへ公開しないでください。

D5a moduleにはchannelごとの出力スイッチがなく、標準bipolar spanは-4 V〜4 Vです: `idle` は `idle_voltage_v` までのrampのみで電気的には切断されず、`turn_on()`/`turn_off()` は非対応、読み出される電圧はmoduleの保持値であり実測値ではありません。

`ramp.rate_v_per_s` × `ramp.step_interval_s` が1 setpointあたりの最大電圧変化です。readback誤差が `readback.tolerance_v` 以内なら設定成功とし、範囲外なら `readback.max_attempts` 回まで再設定します。

`apply_voltage()` は出力を ON にし、現在値 (出力が OFF だった場合はアイドル電圧) から目標値まで ramp します。DC電圧操作は context に紐づき、抜けるとアイドル電圧まで ramp します。

```python
with experiment.dc_voltage_control(mux=6) as dc:
    dc.apply_voltage(0.27)
    state = dc.state
```

`turn_on()` と `turn_off()` は、電圧値を変更せずに選択した mux の出力を ON/OFF します。スイッチを変えるのはこの 2 つの明示操作だけです。

context を抜けた後もバイアスを残す場合は `hold` を選びます。

```python
with experiment.dc_voltage_control(mux=6, on_exit="hold") as dc:
    dc.apply_voltage(0.27)
```

`sweep()` は同じ設定を使って各目標電圧まで順番に ramp します。ramp せずに電圧を印加する必要がある場合だけ `apply_voltage_immediately()` を使います。

### 制御レイアウトの解決規則

`configuration_mode` は固定の channel 数を保証する指定ではなく、優先順を表します。

- `ge-ef-cr` は `ge`、`ef`、`cr` の順に channel を割り当てます。
- `ge-ef-fh` は 3 channel port では `ge`、`ef`、`fh` を割り当て、
  2 channel port では channel 0 に `ge`、channel 1 に `ef` と `fh` を共有します。
- `ge-cr-cr` は `ge`、`cr`、`cr` の順に channel を割り当てます。
- それ以外で control port の channel 数が足りない場合は、左から必要な役割だけを残します。

`quel1se-riken8` では、AWG プロファイルが 4 本の profile-dependent control port を決めます。

- `se8_mxfe1_awg1331` では、これらの port は `1-3-3-1` になります。
  `configuration_mode="ge-ef-cr"` のとき、解決後のレイアウトは
  `ge`、`ge-ef-cr`、`ge-ef-cr`、`ge` です。
- `se8_mxfe1_awg2222` では、これらの port は `2-2-2-2` になります。
  `configuration_mode="ge-ef-cr"` のときは各 port が `ge-ef` に、
  `configuration_mode="ge-ef-fh"` のときは各 port が共有 `ge-ef/fh` に、
  `configuration_mode="ge-cr-cr"` のときは各 port が `ge-cr` に解決されます。

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

skew 測定やクロック同期を使う QuEL-1 system では、`quel1.clock_master` を定義してください。

```yaml
SYSTEM_B:
  chip_id: CHIP_A
  backend: quel1
  quel1:
    clock_master: 10.0.0.10
```

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

Qubex は `wiring.yaml` で `BOX:PORT` と `BOX-PORT` の両方を受け付けますが、保守のしやすさのためにはどちらかに統一することを推奨します。

### `skew.yaml`

箱間タイミングの調整が必要な同期 QuEL-1 / QuBE 構成では、`skew.yaml` を使います。

```yaml
box_setting:
  BOX_A:
    slot: 0
    wait: 0
    port_wait:
      1: 0
  BOX_B:
    slot: 1
    wait: 0
    port_wait:
      8: 0
monitor_port: BOX_A-12
reference_port: BOX_A-1
scale:
  BOX_A-1: 0.125
target_port: !!set
  BOX_A-1: null
  BOX_B-8: null
time_to_start: 0
trigger_nport: 10
```

- `box_setting.<box>.slot` は各 box の粗いタイミング slot を表します。
- `box_setting.<box>.wait` は box 共通の wait 値です。
- `box_setting.<box>.port_wait` は port ごとの差分 wait 値です。
- `reference_port` は基準信号源を選びます。
- `monitor_port` と `trigger_nport` は monitor capture 経路を定義します。
- `target_port` は skew scan に含める port を列挙します。

同じ config を `Experiment` から読み込んだあと、QuEL-1 skew helper で確認と更新ができます。

```python
result = exp.tool.check_skew(["BOX_A", "BOX_B"])
exp.tool.update_skew(250, ["BOX_A", "BOX_B"], backup=True)
result = exp.tool.check_skew(["BOX_A", "BOX_B"])
```

`exp.tool.update_skew(target, ...)` は直前の `check_skew(...)` の推定結果を使い、
各 measured port の実効 wait を `target - measured_idx` だけずらし、
box 共通部分を `wait`、測定した port の差分を `port_wait` に入れてから
`skew.yaml` を上書きします。更新前のファイルを残したい場合は
`backup=True` を指定してください。`skew.yaml.bak.20260520_124900` のような
timestamp 付き backup が作られます。

手順全体は [QuEL-1 skew 調整ワークフロー](../../examples/system/quel1_skew_adjustment.md) を参照してください。

## パラメータファイルを定義する

system 固有の parameter file は、Qubex に渡す `params_dir` の下に置いてください。
推奨形式は parameter family ごとに 1 つの構造化 YAML を置く形式で、旧来の
`params.yaml` と `props.yaml` は互換 fallback として使われます。

認識される file の一覧、読み込み優先度、`control_frequency.yaml` が
`qubit_frequency.yaml` より優先されるような周波数 fallback ルールについては
[パラメータファイル](params-configuration.md) を参照してください。

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
