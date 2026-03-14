# Experiment

`Experiment` は、実機を使う Qubex ユーザーの多くにとって推奨される入口です。
system の設定、装置接続、パルス系列の構築、測定実行、結果解析までを高レベルなワークフローとして提供します。

## Experiment を使うべき人

- 高レベル API を通して量子ビットのパルスレベル実験を進めたい研究者
- 評価、較正、ベンチマークの組み込みワークフローを使いたいユーザー
- セットアップから解析まで一貫した入口を使いたいチーム

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. [システム設定](../getting-started/system-configuration.md) で設定ファイルを用意する
3. [クイックスタート](../getting-started/quickstart.md) で基本ワークフローを確認する
4. [Experiment サンプルワークフロー](examples.md) で notebook をたどる
5. 必要に応じて [コミュニティ提供ワークフロー](../getting-started/contrib-workflows.md) を使う

## 次のような場合は Measurement を選ぶ

- 測定セッションやバックエンド実行にもっと低レベルでアクセスしたい
- 測定スケジュールを直接構築・実行したい
- readout 固有や sequencer レベルのワークフローを扱いたい

その場合は [Measurement 概要](../measurement/index.md) を参照してください。
