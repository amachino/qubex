# qubex の `qubecalib` 依存縮小と `quelware` 直接利用ロードマップ（2026-02-08）

## 1. 背景

- 現状 `qubex` の QuEL backend は `qubecalib` の高レベル API（`QubeCalib`, `sysdb`, `Sequencer`）に強く依存している。
- `quelware_0.10.x` 以降で `qubecalib` API の互換が崩れており、追随コストが増えている。
- 方針として、ハードウェア制御の主軸を `quelware`（`quel_ic_config` + utility群）へ段階的に移す。

## 2. ゴール

- `qubecalib` を必須依存から外し、`qubex` backend が `quelware` の安定 API で動作する状態にする。
- `qubecalib` は移行期間のみ互換レイヤとして任意利用に留める。

## 3. 設計方針

- 依存方向を整理する。
  - 現在: `qubex -> qubecalib -> quelware`
  - 目標: `qubex -> quelware`（必要に応じて薄い adapter）
- `qubex` 内に backend adapter 層を導入し、以下を分離する。
  - Box/Clock 接続管理
  - Port/Channel/Target 構成
  - Sequencer 実行
  - Capture 結果取得
- 「機能単位で差し替え」し、ビッグバン移行は避ける。

## 4. フェーズ別ステップ

### Phase 0: 可観測性の追加（即時）

1. `connect` の実装モードを切替可能にする（legacy / parallel）。
2. 各モードの所要時間を記録し、実機で比較可能にする。
3. 既存挙動を壊さない既定値（legacy）を維持する。

### Phase 1: Box/Clock の直接化

1. `qubecalib.create_box/create_boxpool/read_clock/resync` 依存を廃止。
2. `quel_ic_config`（および `quel_ic_config_utils`）で box 生成・再接続・clock 同期を直接実装。
3. `Quel1BackendController` の clock API を adapter 経由へ移行。

### Phase 2: SystemConfig の置換

1. `qubecalib.sysdb` 依存を段階置換。
2. `qubex` 側で必要最小限の設定モデル（box/port/channel/target relation）を保持。
3. 変換ロジックを `qubecalib` 前提から `quelware` 前提へ更新。

### Phase 3: Sequencer/Execution の置換

1. `Sequencer` 実行パスを抽象化（interface 化）。
2. 既存: `qubecalib` backend executor
3. 新規: `quelware` 直接 executor
4. 実機ベンチで機能・性能・安定性を比較し段階切替。

### Phase 4: 依存整理

1. `qubecalib` を optional dependency 化。
2. CI の必須パスを `quelware` 直接実装に切替。
3. 一定期間後に `qubecalib` adapter を非推奨化・削除。

## 5. 技術的な注意点

- `quelware` の版差分が大きいため、`quelware` 側の LTS/固定バージョン戦略を先に定義する。
- clock/sync 周りは実機依存が強く、ユニットテストだけでは十分でない。
- `qubecalib` は一部で暗黙に再接続・初期化を行うため、直接化時に初期化責務を明確化する必要がある。

## 6. 当面の実装優先順位（推奨）

1. `connect` 系（今回追加した legacy/parallel 切替を起点に計測）。
2. clock 操作（`read_clocks`, `resync_clocks`, `sync_clocks`）。
3. box 操作（`get_box`, `linkup`, `relinkup`, dump APIs）。
4. sequencer 実行。

## 7. 成功判定

- 機能: 既存の主要実験フローが `qubecalib` なしで通る。
- 性能: `connect` と実行時間が現状比で同等以上。
- 運用: backend 障害時の切り戻し手段（adapter 切替）が残っている。
