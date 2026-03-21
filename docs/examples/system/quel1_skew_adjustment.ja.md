# QuEL-1 skew 調整ワークフロー

共有クロックで同期した QuEL-1 / QuBE 構成で、box 間 skew を確認して調整したいときは、この手順を使ってください。

このフローは次を前提にしています。

- `system.yaml` の対象 entry が `quel1` backend を使い、`quel1.clock_master` を定義している
- config ディレクトリに `box.yaml` と `skew.yaml` がある
- `skew.yaml` に monitor 経路、reference port、対象 box が定義されている

## 最小セットアップ

操作対象の QuEL-1 system で `Experiment` を作成します。

```python
import qubex as qx

exp = qx.Experiment(
    system_id="64Q-HF-Q1",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/64Q-HF-Q1",
)
```

有効な `config_dir` には、例えば次のような `skew.yaml` を置きます。

```yaml
box_setting:
  S87R:
    slot: 0
    wait: 0
  S89R:
    slot: 1
    wait: 0
monitor_port: S87R-12
reference_port: S87R-1
scale:
  S87R-1: 0.125
target_port:
  S87R-1: null
  S89R-8: null
time_to_start: 0
trigger_nport: 10
```

`box_setting.*.wait` が skew 調整時に更新する値です。

## 現在の skew を確認する

確認したい box を指定して `exp.tool.check_skew(...)` を実行します。

```python
result = exp.tool.check_skew(["S87R", "S89R"])
```

この helper は次を行います。

- 有効な config ディレクトリから `skew.yaml` と `box.yaml` を読む
- `reference_port` から基準 box を決める
- QuEL-1 backend で現在の skew 測定フローを実行する
- 測定済み skew object と plot を含む `Result` を返す

例えば次のように取り出せます。

```python
skew_runtime = result["skew"]
figure = result.figure
```

## 目標 skew 値を設定する

複数 box に共通の wait 値を入れたいときは、`exp.tool.update_skew(...)` を使います。

```python
exp.tool.update_skew(250, ["S87R", "S89R"], backup=True)
```

この helper は次を行います。

- `skew.yaml` の `box_setting.<box>.wait` を更新する
- `backup=True` なら `skew.yaml.bak` を作る
- 更新後の skew file を有効な QuEL-1 backend に再読込する

`box_ids` を省略すると、`box_setting` にある全 box を更新します。

```python
exp.tool.update_skew(250, backup=True)
```

## 更新後に再確認する

更新したファイルで再度測定します。

```python
result = exp.tool.check_skew(["S87R", "S89R"])
```

返ってきた figure を確認し、必要なら調整を繰り返してください。

収束しない場合の実用上の手順としては、いったん `wait=0` にして確認し、その後で目的の値へ戻すやり方があります。

```python
exp.tool.update_skew(0, ["S87R", "S89R"], backup=True)
exp.tool.check_skew(["S87R", "S89R"])

exp.tool.update_skew(250, ["S87R", "S89R"], backup=True)
exp.tool.check_skew(["S87R", "S89R"])
```

## 関連ページ

- [システム設定](../../user-guide/getting-started/system-configuration.md)
- [`system` サンプルワークフロー](../../user-guide/system/examples.md)
- [`backend` サンプルワークフロー](../../user-guide/backend/examples.md)
