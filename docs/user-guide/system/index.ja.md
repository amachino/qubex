# `system` モジュール

`qubex.system` は、設定読み込みと、1 つの具体的な装置構成を表す
ソフトウェア側モデルを担当するモジュールです。あわせて、
インメモリの `ExperimentSystem` とバックエンドコントローラの状態同期も扱います。

このページは [低レベル API](../low-level-apis/index.md) セクションの一部です。

## `system` を使うべき場面

- 設定ファイルを直接読み込み、その結果できるモデルを確認したい
- `ExperimentSystem`、`QuantumSystem`、`ControlSystem`、target、control parameter を直接扱いたい
- ソフトウェア状態とハードウェア / コントローラ状態を比較・同期したい

## 主要なオブジェクト

- `ConfigLoader`: 選択した 1 つの system を読み込み、`ExperimentSystem` を構築します
- `ExperimentSystem`、`QuantumSystem`、`ControlSystem`: chip、wiring、port、channel、parameter を表すソフトウェア側モデルです
- `SystemManager`: 現在の experiment-system state、backend controller、backend settings を保持する singleton です
- `Quel1SystemSynchronizer` と `Quel3SystemSynchronizer`: `SystemManager` が使う backend 固有の synchronizer です

## 他のモジュールとの関係

- [`measurement`](../measurement/index.md): 読み込まれた system state を使って
  `MeasurementSchedule` や実行フローを組み立てます
- [`backend`](../backend/index.md): `SystemManager` が同期対象にする
  コントローラ状態と実行先ランタイムを提供します

## 推奨する進み方

1. [システム設定](../getting-started/system-configuration.md) で設定ファイルを用意する
2. [低レベル API 概要](../low-level-apis/index.md) で全体像を確認する
3. [`system` サンプルワークフロー](examples.md) から notebook を始める
4. `MeasurementSchedule` や `measurement` の実行フローを扱いたいなら [`measurement`](../measurement/index.md) に進む
5. controller 実装が主題なら [`backend`](../backend/index.md) に進む

## 次のような場合は `Experiment` を選ぶ

- system model を直接扱わずに推奨ワークフローを使いたい
- 設定読み込みや同期処理の詳細を確認する必要がない
- セットアップ、実行、解析まで 1 つの facade で扱いたい
