# `measurement` サンプルワークフロー

このページでは、`measurement` モジュールを起点に進める主要な notebook 入口を紹介します。
`MeasurementSchedule`、キャプチャ/読み出し、sweep などを `measurement` モジュールの流れで扱いたい場合に使ってください。

## 推奨する出発点

- [Measurement config](../../examples/measurement/measurement_config.ipynb): mocked system を含む例で measurement configuration object を構築します。
- [Measurement session](../../examples/measurement/measurement_client.ipynb): measurement session を作成し、実機へ接続して基本測定を行います。
- [Loopback capture](../../examples/measurement/capture_loopback.ipynb): 完全な experiment workflow を使わずに capture の挙動を確認します。

## Sweep と execution のワークフロー

- [Measurement sweep builder](../../examples/measurement/sweep_measurement_builder.ipynb): sweep 設定モデルから `PulseSchedule` を組み立てます。
- [Sweep measurement executor](../../examples/measurement/sweep_measurement_executor.ipynb): `measurement` モジュールの抽象化を使って構造化 sweep workflow を実行します。

## 関連ページ

- [低レベル API 概要](../low-level-apis/index.md)
- [`measurement` モジュール](index.md)
- [`system` モジュール](../system/index.md)
- [`backend` モジュール](../backend/index.md)
- [サンプル集全体](../../examples/index.md)
