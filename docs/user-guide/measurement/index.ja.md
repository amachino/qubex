# Measurement

`Measurement` は、実機ベースの実行をより低レベルで扱うための入口です。
測定セッション、カスタムスケジュール、バックエンド実行、readout 指向のワークフローを直接制御したいときに使います。

## Measurement を使うべき人

- 実機上で測定セッションを直接制御したい上級ユーザー
- バックエンド寄りの実行やスケジュール構築に低レベルでアクセスしたい開発者
- 高レベルな `Experiment` facade の外で、custom readout、sweep、sequencer ワークフローを組みたいユーザー

## Measurement でできること

- 設定読み込み、ハードウェア接続、バックエンド状態確認などの session lifecycle 制御
- measurement schedule の直接実行と custom readout placement
- 構造化された低レベル測定のための sweep builder と execution helper
- readout classification utility とバックエンド固有の execution hook

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. [システム設定](../getting-started/system-configuration.md) で実機設定を用意する
3. [Measurement サンプルワークフロー](examples.md) から notebook を始める

## 次のような場合は Experiment を選ぶ

- 測定内部ではなく、実験中心の高レベルワークフローを使いたい
- 評価、較正、ベンチマークの組み込みルーチンを使いたい
- セットアップ、実行、解析まで 1 つの facade で扱いたい

その場合は [Experiment 概要](../experiment/index.md) を参照してください。
