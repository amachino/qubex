# Measurement

`Measurement` は、実機ベースの実行を measurement 側の概念で扱うための入口です。
`Experiment` も session lifecycle と実行処理では内部的に `Measurement` を使います。
そのため、`Measurement` は `Experiment` より強い入口というより、session、schedule、capture/readout、sweep、backend integration を主語にしたいときの入口です。

このページは [低レベル API](../low-level-apis/index.md) セクションの一部です。

## Measurement を使うべき人

- measurement 側の API やデータモデルを主語に扱いたいユーザー
- backend integration、readout utility、measurement 固有ツールを作る開発者
- `Experiment` 全体の流れよりも、session、schedule、sweep、sequencer の流れを中心に組みたいユーザー

## Measurement でできること

- session lifecycle、schedule 実行、capture/readout、sweep を measurement の語彙で扱う API
- measurement 側のデータモデルや helper への直接アクセス
- readout classification utility とバックエンド固有の execution hook
- `Experiment` が内部で委譲している measurement 基盤

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. [システム設定](../getting-started/system-configuration.md) で実機設定を用意する
3. 必要に応じて共有のパルスシーケンスモデルを確認する: [パルスシーケンスの組み方](../pulse-sequences/index.md)
4. [低レベル API 概要](../low-level-apis/index.md) を確認する
5. [Measurement サンプルワークフロー](examples.md) から notebook を始める

## 次のような場合は Experiment を選ぶ

- 実機実験の多くで推奨されるユーザー向けワークフローを使いたい
- 量子デバイスの特性評価、較正、ベンチマークの組み込みルーチンを使いたい
- measurement 側の語彙を前面に出さず、セットアップ、実行、解析まで 1 つの facade で扱いたい

その場合は [`Experiment` クラス](../experiment/index.md) を参照してください。
