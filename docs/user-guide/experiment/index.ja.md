# `Experiment` クラス

`Experiment` は、実機を使う Qubex ユーザーの多くにとって推奨される入口です。
system の設定、装置接続、パルスシーケンスの構築、測定実行、結果解析までを高レベルなワークフローとして提供します。

## Experiment を使うべき人

- 高レベル API を通して量子ビットのパルスレベル実験を進めたい研究者
- 量子デバイスの特性評価、較正、ベンチマークの組み込みワークフローを使いたいユーザー
- セットアップから解析まで一貫した入口を使いたいチーム

## 推奨する進み方

1. [インストール](../getting-started/installation.md) で Qubex を入れる
2. [システム設定](../getting-started/system-configuration.md) で設定ファイルを用意する
3. [クイックスタート](../getting-started/quickstart.md) で基本ワークフローを確認する
4. [Experiment サンプルワークフロー](examples.md) で notebook をたどる
5. 必要に応じて [コミュニティ提供ワークフロー](../getting-started/contrib-workflows.md) を使う

## Experimental な非同期 API

`Experiment` には、async-first なメソッドも公開されています。

- `run_measurement()`
- `run_sweep_measurement()`
- `run_ndsweep_measurement()`

これらは Experimental な機能として扱ってください。公開 API ではありますが、
非同期ワークフローの整理が続いているため、将来のリリースでシグネチャ、
挙動、結果の扱いが変わる可能性があります。

現時点で API 安定性を優先するなら、従来の同期メソッド
（`measure()`、`execute()`、`sweep_parameter()`）を推奨します。

## 次のような場合は 低レベル API を選ぶ

- `MeasurementSchedule`、キャプチャ/読み出し、sweep など、`measurement` モジュールの実行フローを直接扱いたい
- 設定読み込み、`ExperimentSystem`、同期処理を直接扱いたい
- backend integration、controller レベルの execution path、QuEL 固有 runtime tooling を扱いたい

その場合は [`measurement`、`system`、`backend` を整理した低レベル API 概要](../low-level-apis/index.md) を参照してください。
