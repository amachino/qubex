# quelware 0.8系 から 0.10系 への移行ガイド（2026-02-08）

## 1. 対象と目的

- 対象: `quelware v0.8.x`（例: `v0.8.14`）を利用している `qubex` backend。
- 目的: `quelware v0.10.x` へ安全に移行し、運用を継続できる状態を作る。

## 2. 重要な差分（先に押さえる点）

1. API互換

- `qubecalib` 側で互換破壊がある（トップレベル export 変更、一部メソッド削除）。
- `qubex` が `qubecalib` 旧APIを直接使っている場合、そのままでは動かない箇所が出る。

2. 依存構造

- 0.8系時代の `quel_clock_master` サブパッケージ前提コードは見直しが必要。
- 0.10系では `quel_ic_config` 周辺が強化・再編されている。

3. ビルド/環境

- 0.10系周辺では GUI/描画系依存（`PyGObject` / `pycairo`）が絡むパスがあり、環境差分で導入失敗しやすい。

## 3. 推奨移行戦略（段階移行）

### Step 1: 接続処理を切替可能にする

- `legacy`（現行）と `parallel/new`（新実装）をオプションで選べるようにする。
- 同一設定・同一boxで所要時間を計測し、差分を可視化する。
- 本番既定値は最初 `legacy` のままにして、切戻し可能にする。

### Step 2: Box/Clock API の依存を薄くする

- `create_boxpool/read_clock/resync` など `qubecalib` 直依存箇所を adapter に隔離。
- adapter の裏側で `quelware` 直接実装を追加し、切替で比較する。

### Step 3: Sequencer 実行経路を二重化

- 既存経路: `qubecalib` executor
- 新経路: `quelware` 直接 executor
- 同一入力で結果（status/data/config）と時間を比較する。

### Step 4: 既定値を 0.10系に寄せる

- CI を 0.10系でグリーン化。
- 実機の回帰項目（後述）を継続監視。
- 問題がなければ既定値を `parallel/new` に切替。

## 4. 回帰テスト観点（必須）

1. 接続

- 複数 box 接続成功
- reconnect/relinkup 成功
- 失敗時リトライ/例外メッセージ

2. clock

- read/sync/resync の成功率
- 2台以上で timestamp 整合

3. 実行

- AWG/Capture 初期化
- sequencer 実行結果整合
- 測定値の統計（平均/分散）差分が許容範囲内

4. 性能

- connect 時間（legacy vs new）
- 典型実験 1-run あたり時間

## 5. ロールバック方針

- 1フラグで `legacy` 実装に戻せる状態を維持する。
- 版固定（lock file / commit hash）を徹底し、環境再現手順を明文化する。
- 本番切替は段階的（開発 -> 検証機 -> 本番機）で実施する。

## 6. 推奨運用ルール

- 0.10系へ移行完了まで、`qubecalib` と `quelware` の更新を同時に行わない。
- 1回の変更で「接続」「clock」「実行」のうち1領域だけを変更する。
- 実機回帰の記録（成功率、時間、エラーログ）を毎回残す。

## 7. 今回の qubex への適用状況

- `Quel1BackendController.connect()` は `legacy` / `parallel` 切替可能。
- `MeasurementBackendManager.connect()` から `parallel` フラグが伝播される。
- 速度比較のための最小基盤は整備済み。

## 8. 次アクション（推奨）

1. 実機で `parallel=False` と `parallel=True` の connect 時間を比較する。
2. 失敗ケース（1台リンクダウン、clock未同期）で挙動差を確認する。
3. 問題なければ、clock API の直接 `quelware` 化に着手する。
