# Simulator

`Simulator` は、パルスレベルのハミルトニアン解析をオフラインで行うための入口です。
量子系をモデル化し、pulse を与え、実機に接続せずに実験を反復したいときに使います。

## Simulator を使うべき人

- 実機を使わずにパルスレベルのダイナミクスを調べたい研究者
- 実システムへ移る前にモデルの振る舞いを確かめたいユーザー
- 較正やパルス設計をオフラインで試したいチーム

## Simulator でできること

- qubit、resonator、結合系に対するパルスレベルのハミルトニアンシミュレーション
- Qubex の pulse object をオフライン解析にそのまま再利用
- 実機時間を使う前に較正フローを試す安全な経路

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. [Simulator サンプルワークフロー](examples.md) から notebook を始める

Simulator notebook を始めるのに、実機向けの設定ファイルは不要です。

## 次のような場合は Experiment を選ぶ

- 実機上で実験を実行したい
- measurement result や実機ベースの readout が必要
- 接続、実行、解析まで含む高レベルワークフローを使いたい

その場合は [Experiment 概要](../experiment/index.md) を参照してください。
