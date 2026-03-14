# Measurement サンプルワークフロー

このページでは、`Measurement` ワークフローの主要な notebook 入口を紹介します。
`Experiment` よりも直接的に実機実行を制御したい場合に使ってください。

## 推奨する出発点

- [Measurement config](../../examples/measurement/measurement_config.ipynb): mocked system を含む例で measurement configuration object を構築します。
- [Measurement session](../../examples/measurement/measurement_client.ipynb): measurement session を作成し、実機へ接続して基本測定を行います。
- [Loopback capture](../../examples/measurement/capture_loopback.ipynb): 完全な experiment workflow を使わずに capture の挙動を確認します。

## Sweep と execution のワークフロー

- [Measurement sweep builder](../../examples/measurement/sweep_measurement_builder.ipynb): sweep 設定モデルから `PulseSchedule` を組み立てます。
- [Sweep measurement executor](../../examples/measurement/sweep_measurement_executor.ipynb): measurement 側の抽象化を使って構造化 sweep workflow を実行します。
- [QuEL-3 sequencer builder flow](../../examples/measurement/quel3_sequencer_builder_flow.ipynb): バックエンド固有の低レベル sequencer 経路を扱います。

## 関連ページ

- [Measurement 概要](index.md)
- [サンプル集全体](../../examples/index.md)
