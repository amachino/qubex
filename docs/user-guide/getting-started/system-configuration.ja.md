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
    step_size_v: 0.01
    wait_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  reset_voltage: 0.0
  overrides:
    - mux: 7
      ramp:
        rate_v_per_s: 0.05
```

- `devices` — `driver` が adapter を選び、`channels` がデバイスの出力を列挙します。`params` の中身は driver 固有で、選択した driver 自身が検証します (ONS61797: serial は `port`、network は `ip_address`。両方は不可)。Qubex は ONS61797 の書き込みを 0 V〜4 V、channel を 1〜16 に制限し、独立制御モード (`OMD 0`) を必須とします。Qblox driver は -4 V〜4 V を許可します。
- `wiring` — 各エントリは役割名 (`bias`) と `デバイス名-チャンネル` 形式の出力参照 (`ONS1-2` = デバイス `ONS1` の channel 2、1 始まり) を持ちます。操作できるのは配線済みの出力だけで、未配線の mux や `channels` 外のチャンネルは推測せず明示エラーになります。
- `settings` — 直下の値が配線済み全 mux の既定値、`overrides` が mux 単位の上書きです。1 つの `settings` (役割は既定 `bias`、`role` で変更可) が拾う出力はすべて同じデバイスにある必要があります。

アイドル電圧 (DC bias を使用していない間、各出力を保持する電圧) は較正値で、`jpa_params.yaml` の `idle_voltage` に置きます (未較正の mux は `reset_voltage` へフォールバック)。DC 電圧 context は電圧印加・sweep・readbackの間に1つのデバイス接続を保持し、終了時に必ずアイドル電圧まで ramp してから接続を閉じます。出力スイッチは電圧印加時に暗黙には切り替わらず、`reset_dc_voltages()` と `shutdown_dc_voltages()` で明示的に管理します。

直接接続の `ons61797` driverはcontextの間、装置接続を排他的に所有します。
同じ装置の別channelを別contextや別processから同時に開かないでください。
複数clientでの利用は、将来追加するNF向けserver driverで対応する予定です。

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
    step_size_v: 0.01
    wait_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  reset_voltage: 0.0
```

serverは各出力を `<デバイス名>-<channel>` で識別するため、デバイス名はbackendの報告名に合わせます (`Qblox1` → `Qblox1-15`)。命名が規則的でない場合は `params` の `device_names: {channel: 名前}` で対応します。rampはserver側で一括実行されるため他clientは割り込めませんが、完了まで他channelの操作が待つことがあります。認証のないsocketを信頼できないnetworkへ公開しないでください。

D5a moduleにはchannelごとの出力スイッチがなく、標準bipolar spanは-4 V〜4 Vです: `idle` と `shutdown` は指定電圧までのrampのみで電気的には切断されず、読み出される電圧はmoduleの保持値であり実測値ではありません。

`ramp` の各値は backend sweep と同じ語彙で、そのまま backend へ渡されます: `rate_v_per_s` が全体の速さ (所要時間 ≈ |ΔV| / rate)、`step_size_v` が setpoint の電圧刻み、`wait_s` が 1 setpoint の最小滞在時間です。readback誤差が `readback.tolerance_v` 以内なら設定成功とし、範囲外なら `readback.max_attempts` 回まで再設定します。

Qblox固有のrate・step・wait制約は、channelを変更する前の設定load時に検証します。ソフトウェア生成するONS61797のrampは、中間setpointごとの通信を避け、最終目標だけreadback検証します。

`apply_voltage()` は ON 状態の出力を現在値から目標値まで ramp します (OFF なら暗黙に ON 化せずエラー)。DC電圧操作は context に紐づき、抜けるとアイドル電圧まで ramp します。

```python
with experiment.external_devices.dc_voltage(mux=6) as dc:
    dc.apply_voltage(0.27)
    state = dc.state
```

電圧印加は出力が ON であることが前提で、OFF なら暗黙に ON 化せずエラーになります。`reset_dc_voltages()` は選択した mux を「出力 ON・リセット電圧」の既知状態に揃えます。保守・チップ交換・安全停止時に使う `shutdown_dc_voltages()` はリセット電圧まで ramp して、対応機器では物理出力を OFF にします。通常の実験contextはshutdownせずidleへ戻ります。`get_dc_voltage_states()` はactiveな配線済み全muxを1接続でまとめて読み、`bias_dc_voltages()` は較正済み mux を bias 電圧へ、`idle_dc_voltages()` はアイドル電圧へ ramp します。box 系 API と同じく任意の `muxes` 引数 (index またはラベル、省略時はactiveな配線済み全mux) で対象を絞れます。いずれも書き込み操作なので、box への push と同様に実行前へ確認プロンプトが出ます。各一括書き込みと、その直後のreadbackは1つのデバイス接続を共有します。

選択が空、または確認を拒否した場合は、装置へ接続せず `{}` を返します。

一時的な context を使わず、較正済みのactiveな配線済み全muxを bias 電圧へ設定する場合は一括操作を使います。

```python
experiment.external_devices.bias_dc_voltages()
```

`sweep()` は同じ設定を使って各目標電圧まで順番に ramp します。

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
