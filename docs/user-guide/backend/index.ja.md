# `backend` モジュール

`qubex.backend` は、共通の controller 契約と、実機実行を担う
QuEL-1 / QuEL-3 の具体実装を定義するモジュールです。低レベル API の最下層に
位置し、主にバックエンド統合、ランタイム検証、backend 固有の実行経路を扱います。

このページは [低レベル API](../low-level-apis/index.md) セクションの一部です。

## `backend` を使うべき場面

- backend controller を実装または検証したい
- `BackendExecutionRequest`、backend result payload、backend kind を直接扱いたい
- QuEL 固有の deploy、sequencer、execution path を扱いたい

## 主要なオブジェクト

- `BackendController`、`BackendExecutionRequest`、`BackendKind`: 共通の controller 契約です
- `Quel1BackendController` と `Quel3BackendController`: サポート対象 backend family の具体実装です
- `Quel1ExecutionPayload`、`Quel3ExecutionPayload`、`Quel3SequencerBuilder` などの backend 固有 model / builder
- `qubex.measurement.adapters`: measurement schedule / config から backend request へ橋渡しする層です
- [`start_continuous_wave()`](continuous-wave.md) など、hardware-level CW 出力のための
  QuEL-1 optional controller capability

## 直接利用は上級者向けです

実機ワークフローの多くは `Experiment` か
[`measurement`](../measurement/index.md) から始めるのが適切です。
`backend` を直接使うのは、controller レベルの挙動そのものが主題のときに限るのが適切です。

## QuEL-3 の実行セッション

実行ログには、session を開いた時点の session ID とリクエストの試行番号を
`INFO` で記録します。リトライと後始末の失敗は `WARNING`、最終的なリクエストの
失敗は session ID と traceback 付きの `ERROR` で記録します。

原因候補は `qubex.backend.quel3.managers.session_workarounds` の
`QUELWARE_EXCEPTION_HINTS` に追加できます。キーにはモジュール名を含む例外クラス名
（例: `quelware_client.core.exceptions.LockConflictError`）、値には表示したい文面を
指定します。初期状態は空です。登録した文面は失敗ログに `possible cause` として表示し、
サブクラスや明示的に連鎖した原因例外にも適用します。リトライの判断や例外は変更しません。

session 作成では、既知の resource / unit 利用不可を待機時間を延ばしながら最大4回
試行します。外側では `Exception` が発生した payload を client / session を作り直して
最大4回試行し、その各試行で session 作成の試行枠を使います。キャンセルは再試行しません。
trigger 後の失敗では同じ payload が実機で再実行される場合があります。
正常な client はバッチ内で再利用し、最後の後始末の失敗は結果や元の例外を上書きせず
ログに記録します。

## 推奨する進み方

1. [低レベル API 概要](../low-level-apis/index.md) で全体像を確認する
2. schedule や result から話を始めるなら、先に [`measurement`](../measurement/index.md) を読む
3. QuEL-1 の CW 確認では [QuEL-1 連続波出力](continuous-wave.md) を読む
4. [`backend` サンプルワークフロー](examples.md) に進む
5. controller の詳細は [API リファレンス](../../api-reference/qubex/backend/index.md) を参照する

## 次のような場合は別のモジュールを選ぶ

- [`system`](../system/index.md): 設定読み込み、インメモリ model、同期処理が主題
- [`measurement`](../measurement/index.md): `MeasurementSchedule`、キャプチャ/読み出し、sweep、`measurement` の実行フローが主題

## 次のような場合は `Experiment` を選ぶ

- 実機実験を進めるための推奨ワークフローを使いたい
- controller レベルの実行詳細を確認する必要がない
- セットアップ、実行、解析まで 1 つの facade で扱いたい
