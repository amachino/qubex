# `backend` モジュール

`qubex.backend` は、共通の controller 契約と、実機実行を担う
QuEL-1 / QuEL-3 の具体実装を定義するモジュールです。低レベル API の最下層に
位置し、主にバックエンド統合、ランタイム検証、backend 固有の実行経路を扱います。

このページは [低レベル API](../low-level-apis/index.md) セクションの一部です。

## `backend` を使うべき場面

- backend controller を実装または検証したい
- `BackendExecutionRequest`、backend result payload、backend kind を直接扱いたい
- QuEL 固有の deploy、sequencer、execution path を扱いたい

## 主要なオブジェクト

- `BackendController`、`BackendExecutionRequest`、`BackendKind`: 共通の controller 契約です
- `Quel1BackendController` と `Quel3BackendController`: サポート対象 backend family の具体実装です
- `Quel1ExecutionPayload`、`Quel3ExecutionPayload`、`Quel3SequencerBuilder` などの backend 固有 model / builder
- `qubex.measurement.adapters`: measurement schedule / config から backend request へ橋渡しする層です

## 直接利用は上級者向けです

実機ワークフローの多くは `Experiment` か
[`measurement`](../measurement/index.md) から始めるのが適切です。
`backend` を直接使うのは、controller レベルの挙動そのものが主題のときに限るのが適切です。

## 推奨する進み方

1. [低レベル API 概要](../low-level-apis/index.md) で全体像を確認する
2. schedule や result から話を始めるなら、先に [`measurement`](../measurement/index.md) を読む
3. [`backend` サンプルワークフロー](examples.md) に進む
4. controller の詳細は [API リファレンス](../../api-reference/qubex/backend/index.md) を参照する

## 次のような場合は別のモジュールを選ぶ

- [`system`](../system/index.md): 設定読み込み、インメモリ model、同期処理が主題
- [`measurement`](../measurement/index.md): session、schedule、capture/readout、sweep が主題

## 次のような場合は `Experiment` を選ぶ

- 実機実験を進めるための推奨ワークフローを使いたい
- controller レベルの実行詳細を確認する必要がない
- セットアップ、実行、解析まで 1 つの facade で扱いたい
