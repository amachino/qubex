# quelware 0.8系 → 0.10系 移行 実行手順書（チェックリスト付き）

## 0. 目的

- `qubex` を `quelware 0.8系` から `0.10系` へ、安全に段階移行する。
- 既存運用を止めずに、性能比較・互換確認・切戻しまで一連で実施する。

## 1. 参照ドキュメント

- 調査レポート: `/Users/amachino/qubex/docs/reports/qubecalib-quelware-0.10x-migration-report-2026-02-08.md`
- 直接化ロードマップ: `/Users/amachino/qubex/docs/reports/quelware-direct-migration-roadmap-2026-02-08.md`
- 0.8→0.10移行ガイド: `/Users/amachino/qubex/docs/reports/quelware-0.8-to-0.10-migration-guide-2026-02-08.md`

## 2. 事前条件

- 現在ブランチと依存バージョンを記録済み。
- 実機アクセス可能（対象 box 一覧が確定）。
- 切戻し手段（`legacy` モード）が有効。
- 今回追加済み機能:
  - `Quel1BackendController.connect(..., parallel=...)`

## 3. 実行フェーズ

### Phase A: 準備

1. 変更前状態を固定する。
2. テストを通して基準状態を確定する。
3. 実機対象 box リストを確定する。

### Phase B: connect速度比較（legacy vs parallel）

1. 同一 box リストで `parallel=False` を複数回実行。
2. 同一 box リストで `parallel=True` を複数回実行。
3. 実行ログや外部計測で所要時間を比較する。
4. 失敗率・再試行回数・ログを併記する。

### Phase C: 機能回帰

1. linkup/relinkup。
2. read_clocks/check_clocks/resync_clocks。
3. 代表的な sequencer 実行と結果整合。

### Phase D: 段階切替

1. 検証機で `parallel=True` 常用。
2. 問題なければ本番で段階適用。
3. 異常時は即 `parallel=False` へ戻す。

### Phase E: 直接quelware化準備

1. `qubecalib` 経由 API を adapter に集約。
2. Box/Clock から `quelware` 直接経路を先行導入。
3. sequencer 経路は最後に置換。

## 4. チェックリスト

### 4.1 準備チェック

- [ ] 現在の commit hash を記録した。
- [ ] `uv run pytest` の基準結果を保存した。
- [ ] 対象 box ID と台数を確定した。
- [ ] 異常時の連絡経路を確認した。

### 4.2 速度比較チェック

- [ ] `parallel=False` を最低3回実行した。
- [ ] `parallel=True` を最低3回実行した。
- [ ] 各回で `mode/box_count/boxpool/quel1system/total` を記録した。
- [ ] 平均値と最大値を比較した。
- [ ] 失敗・タイムアウトの有無を記録した。

### 4.3 機能回帰チェック

- [ ] connect 後に box が期待台数で見える。
- [ ] link_status が全ポートで期待通り。
- [ ] clocks の同期判定が期待通り。
- [ ] resync 後に同期が回復する。
- [ ] 代表実験1件以上で実行成功。
- [ ] 代表実験の主要指標が許容範囲内。

### 4.4 切替判定チェック

- [ ] `parallel=True` が `legacy` と同等以上の成功率。
- [ ] `parallel=True` が性能上の明確な不利を示さない。
- [ ] 既知不具合と回避策を記録した。
- [ ] 本番適用の Go/No-Go 判定を明文化した。

### 4.5 ロールバックチェック

- [ ] `parallel=False` へ即時切替できる。
- [ ] 切戻し手順を運用担当が実行できる。
- [ ] 切戻し後の健全性確認項目を定義済み。

## 5. 記録テンプレート（そのまま利用可）

## 実行情報

- 日時:
- 実行者:
- 環境:
- 対象 box:
- commit:

## connect計測

- trial1: mode=legacy, box_count=, boxpool=, quel1system=, total=
- trial2: mode=legacy, box_count=, boxpool=, quel1system=, total=
- trial3: mode=legacy, box_count=, boxpool=, quel1system=, total=
- trial1: mode=parallel, box_count=, boxpool=, quel1system=, total=
- trial2: mode=parallel, box_count=, boxpool=, quel1system=, total=
- trial3: mode=parallel, box_count=, boxpool=, quel1system=, total=

## 回帰結果

- linkup/relinkup:
- clock read/resync:
- sequencer実行:
- 指標差分:

## 判定

- Go/No-Go:
- 根拠:
- 残課題:

## 6. 冗長処理メモ（現時点）

- `connect` 直後に `create_resource_map("cap")` と `create_resource_map("gen")` を毎回再構築している。
  - box/target 構成が不変なら差分更新またはキャッシュ検討余地あり。
- `system_manager.pull(...)` と `connect(...)` の責務が一部重なっている可能性がある。
  - 実運用ログを見て、重複 I/O（同一情報の再取得）がないか確認推奨。

## 7. 次アクション

1. 本手順で実機ベンチを1サイクル実施。
2. 記録テンプレートを埋めて共有。
3. 結果が良ければ Phase E（直接quelware化）へ進む。
