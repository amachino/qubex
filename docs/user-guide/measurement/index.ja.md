# `measurement` モジュール

`qubex.measurement` は、実機ベースの実行を measurement 側の概念で扱う
モジュールです。[`system`](../system/index.md) と
[`backend`](../backend/index.md) のあいだに位置し、読み込まれた
システム状態を使って schedule から capture/readout や sweep の実行フローを
組み立てます。

このページは [低レベル API](../low-level-apis/index.md) セクションの一部です。

## `measurement` を使うべき場面

- session、schedule、capture/readout、sweep、測定結果を主語にしたい
- `Measurement`、`MeasurementSchedule`、`SweepMeasurementExecutor` を直接使いたい
- backend 固有 controller に落ちる前の、backend 非依存な実行フローを扱いたい

## 主要なオブジェクト

- `Measurement`: session lifecycle と実行処理の facade
- `MeasurementSchedule`、`MeasurementResult`、sweep result 系: measurement 側の標準モデル
- builder / executor: `MeasurementScheduleBuilder`、`SweepMeasurementBuilder`、`SweepMeasurementExecutor`
- service / adapter: session / execution / classification service と `MeasurementBackendAdapter` 実装

## 他のモジュールとの関係

- [`system`](../system/index.md): `ConfigLoader`、`ExperimentSystem`、
  target、parameter state など、`Measurement` が依存する状態を提供します
- [`backend`](../backend/index.md): measurement adapter が最終的に接続する
  controller 契約と、QuEL-1 / QuEL-3 の具体実装を提供します

## 推奨する進み方

1. [低レベル API 概要](../low-level-apis/index.md) で全体像を確認する
2. パルスシーケンスから始まる場合は [パルスシーケンスの組み方](../pulse-sequences/index.md) を読む
3. [`measurement` サンプルワークフロー](examples.md) から notebook を始める
4. 設定読み込みや同期が主題なら [`system`](../system/index.md) に進む
5. controller や payload が主題なら [`backend`](../backend/index.md) に進む

## 次のような場合は `Experiment` を選ぶ

- 実機実験の多くで推奨されるユーザー向けワークフローを使いたい
- 量子デバイスの特性評価、較正、ベンチマークの組み込みルーチンを使いたい
- measurement 側の語彙を前面に出さず、セットアップ、実行、解析まで 1 つの facade で扱いたい

その場合は [`Experiment` クラス](../experiment/index.md) を参照してください。
