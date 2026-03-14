# 低レベル API

このセクションは、`Experiment` の下でモジュール単位に制御したい開発者向けです。
構成は `measurement`、`system`、`backend` の 3 つのモジュールを軸にしています。

Qubex の多くの利用者は、実機では [`Experiment` クラス](../experiment/index.md)、
オフラインでは [`QuantumSimulator` クラス](../simulator/index.md) から始めるのが適切です。
session、システムモデル、backend controller を主役にしたいときに、
このセクションを使ってください。

## モジュール対応表

| モジュール | 担当する責務 | こんなときに使う |
| --- | --- | --- |
| [`measurement`](../measurement/index.md) | session、schedule、capture/readout、sweep、result 変換 | measurement 側の実行フローを直接組みたい |
| [`system`](../system/index.md) | 設定読み込み、システムモデル、ソフトウェアとハードウェアの状態同期 | 1 つの system 定義を読み込み、状態を確認・同期したい |
| [`backend`](../backend/index.md) | backend controller の共通契約と QuEL 固有実装 | controller レベルの実行や backend 固有 payload を扱いたい |

## モジュール同士の関係

1. [`system`](../system/index.md) が設定ファイルを読み込み、ソフトウェア側の
   `ExperimentSystem` を組み立てます。
2. [`measurement`](../measurement/index.md) がその状態を使って、session、
   schedule、capture/readout、sweep のフローを構築します。
3. [`backend`](../backend/index.md) の controller が、準備された request を
   実際の QuEL ランタイムへ流します。

## 推奨される導線

- [`measurement`](../measurement/index.md): session lifecycle、
  `MeasurementSchedule`、capture/readout、sweep を扱うならここから始め、
  続けて [`measurement` サンプルワークフロー](../measurement/examples.md) に進みます。
- [`system`](../system/index.md): `ConfigLoader`、`ExperimentSystem`、
  `SystemManager`、同期処理を扱うならここから始め、
  続けて [`system` サンプルワークフロー](../system/examples.md) に進みます。
- [`backend`](../backend/index.md): `BackendController`、backend kind、
  QuEL 固有実装を扱うならここから始め、
  続けて [`backend` サンプルワークフロー](../backend/examples.md) に進みます。

## 次のような場合は `Experiment` を選ぶ

- 実機実験の多くで推奨されるユーザー向けワークフローを使いたい
- 量子デバイスの特性評価、較正、ベンチマークの組み込みルーチンを使いたい
- セットアップ、実行、解析までを 1 つの facade で扱いたい

## 関連ドキュメント

- [パルスシーケンスの組み方](../pulse-sequences/index.md)
- [サンプル集](../../examples/index.md)
- [API リファレンス](../../api-reference/qubex/index.md)
