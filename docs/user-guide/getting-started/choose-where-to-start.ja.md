# 始め方を選ぶ

Qubex のドキュメントは、2 つの推奨ワークフロー、1 つの共有概念セクション、
1 つの専門 API セクションで構成されています。
多くの利用者は `Experiment` か `Simulator` から始めるのが適切です。

## まずはここから

| 開始点 | こんなときに選ぶ | 次に見るページ |
| --- | --- | --- |
| `Experiment` | 実機実験を、セットアップから解析まで含む推奨ワークフローで進めたい | [Experiment 概要](../experiment/index.md) |
| `Simulator` | 実機に接続せず、パルスレベルのダイナミクスをオフラインで扱いたい | [Simulator 概要](../simulator/index.md) |

## 共有の pulse 系列モデルを先に学ぶ

実機で流すか、オフラインで使うかを決める前に、Qubex で pulse 系列をどう組むか
知りたい場合は `パルスシーケンス` から入ります。

推奨する進み方:

- [PulseSchedule の組み方](../pulse-sequences/index.md)
- [Pulse tutorial notebook](../../examples/pulse/tutorial.ipynb)

## 必要なときだけ低レベル API を使う

session、schedule、capture/readout、sweep、backend 実行の詳細を主語にしたい場合は、
`低レベル API` に進みます。
このセクションの中心にある `Measurement` は、`Experiment` における session
lifecycle と measurement execution を支える基盤でもあります。

推奨する進み方:

- [低レベル API 概要](../low-level-apis/index.md)
- [Measurement API 概要](../measurement/index.md)
- [Measurement サンプルワークフロー](../measurement/examples.md)

## セットアップに関する補足

- まず Qubex をインストールします: [インストール](installation.md)
- `Experiment` と実機ベースの `低レベル API` では設定ファイルと parameter ファイルが必要です: [システム設定](system-configuration.md)
- `Simulator` は実機向け設定ファイルを必要としません

## 利用者ではなく開発者として参加する

Qubex のコードベース自体を拡張したい場合は、まず [Contributing](../../CONTRIBUTING.md) を読み、その後で [開発ガイド](../../developer-guide/index.md) に進んでください。
