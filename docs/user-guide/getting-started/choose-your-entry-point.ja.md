# 入口を選ぶ

Qubex には 3 つの主要な入口があります。必要な抽象度と実行環境に合わせて選んでください。

## 早見表

| 入口 | こんなときに選ぶ | 開始ページ |
| --- | --- | --- |
| `Experiment` | 実機上で量子ビットのパルスレベル実験を高レベルなワークフローで進めたい | [Experiment 概要](../experiment/index.md) |
| `Measurement` | ハードウェア接続された測定セッション、カスタムスケジュール、低レベル readout フローを直接制御したい | [Measurement 概要](../measurement/index.md) |
| `Simulator` | 実機を使わずにパルスレベルのハミルトニアンダイナミクスをオフラインで扱いたい | [Simulator 概要](../simulator/index.md) |

## 共通の前提条件

- まず Qubex をインストールします: [インストール](installation.md)
- `Experiment` と `Measurement` では、system 用の設定ファイルと parameter ファイルが必要です: [システム設定](system-configuration.md)

## Experiment

接続、構成、パルス生成、実行、解析まで、研究でよく使うワークフローを Qubex にまとめて扱わせたい場合は `Experiment` を選びます。

推奨する進み方:

- [Experiment 概要](../experiment/index.md)
- [クイックスタート](quickstart.md)
- [Experiment サンプルワークフロー](../experiment/examples.md)

## Measurement

実機上で測定セッション、バックエンド実行、スケジュール構築、readout 固有の処理をより直接制御したい場合は `Measurement` を選びます。

推奨する進み方:

- [Measurement 概要](../measurement/index.md)
- [Measurement サンプルワークフロー](../measurement/examples.md)

## Simulator

実機に接続せず、パルスレベルのモデル、較正、ダイナミクスを反復したい場合は `Simulator` を選びます。

推奨する進み方:

- [Simulator 概要](../simulator/index.md)
- [Simulator サンプルワークフロー](../simulator/examples.md)

## 利用者ではなく開発者として参加する

Qubex のコードベース自体を拡張したい場合は、まず [Contributing](../../CONTRIBUTING.md) を読み、その後で [開発ガイド](../../developer-guide/index.md) に進んでください。
