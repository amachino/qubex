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
  backend: quel1
  quel1:
    clock_master: 10.0.0.10
```

- トップレベルのキーが `system_id` です。
- `backend` で、この system が使うバックエンド種別を選びます。
- バックエンド固有セクションの名前は `backend` と同じにします。

skew 測定やクロック同期を使う QuEL-1 system では、
`quel1.clock_master` に clock master の IP address を指定してください。

#### QuEL-3 を使用する場合

QuEL-3 を使用する場合は、`backend` に `quel3` を指定し、`quel3` セクションに
endpoint の接続情報を定義します。

```yaml
SYSTEM_QUEL3:
  chip_id: CHIP_A
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
    transport: grpc
```

QuEL-3 では、`transport` を省略するか `grpc` にすると native gRPC を使います。

#### クラウド公開された QuEL-3 に接続する場合

さらに、Cloudflare Access で保護してクラウド公開した QuEL-3 に接続する場合は、
`transport` に `https` を指定します。secret header ごとに mount した secret file
の path を指定し、header の値を `system.yaml` に直接保存しないでください。

```yaml
SYSTEM_CLOUDFLARE:
  chip_id: CHIP_A
  backend: quel3
  quel3:
    endpoint: api.example.com
    # port: 443
    transport: https
    http_transport:
      # base_path: /your-api-base-path
      default_timeout_seconds: 30.0
      proxy:
        url_path: /run/secrets/quelware-proxy-url
      secret_header_paths:
        CF-Access-Client-Id: /run/secrets/cf-access-client-id
        CF-Access-Client-Secret: /run/secrets/cf-access-client-secret
```

`https` transport では、`port` を省略すると `443` を使用します。
`http_transport.base_path` を指定しない場合、request は endpoint のルートへ
送信されます。`http_transport.default_timeout_seconds` は HTTP request の開始から
response body の読み取り完了までを秒単位で制限します。transport 全体の
デフォルト値が不要なら省略してください。

`secret_header_paths` は HTTPS でのみ使用できます。key には HTTP header 名、
value にはその header の secret 値を保存した file の path を指定します。
transport は指定された header を request ごとに送信します。Cloudflare Access を
使用する場合は、application 側に対応する Service Auth policy を設定してください。
HTTPS certificate は system trust store を使って検証されます。redirect は
scheme、host、実効 port が同じ場合にのみ追跡されます。cross-origin redirect は、
request credential を redirect 先へ送信する前に失敗します。

`proxy` を省略した場合、`https` transport は標準の `HTTP_PROXY`、
`HTTPS_PROXY`、`NO_PROXY` 環境変数を使用します。system ごとに proxy を明示
する場合は、完全な HTTP URL を `proxy.url_path` で指定したファイルへ保存
してください。明示 URL は `HTTP_PROXY` または `HTTPS_PROXY` より優先されます
が、endpoint が `NO_PROXY` に一致する場合は直接接続します。

Basic 認証付き proxy では、
`http://user:password@proxy.example.com:3128` のような URL をファイルへ保存
します。username や password の予約文字は percent-encode し、ファイルを
secret として保護してください。URL を `system.yaml` へ直接書かないでください。
SOCKS、NTLM、Kerberos proxy 認証および HTTPS proxy URL には対応しません。
TLS を検査する proxy が private CA を使う場合は、その CA を system trust
store へインストールしてください。

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
