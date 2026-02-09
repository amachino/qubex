# 旧来装置 + quel-3 併用時の実行手順書（チェックリスト付き）

## 0. 対象

- 旧来装置（`qubecalib` 経由）
- 新規 `quel-3` 装置（`quelware-client` 経由）

## 1. 実行方針

- 装置ごとに backend controller を選択する。
- 実験ジョブ開始時に対象resource/targetから controller を決定し、
  同一controller内で session/connect をまとめる。

## 2. 事前準備

1. `box.yaml` に backend 種別情報を追加。
2. `controller_endpoint`（quelware server）疎通確認。
3. Python実行環境が `quelware-client` 要件（>=3.12）を満たすか確認。
4. 切戻し時の legacy-only 実行手段を確認。

## 3. 実行手順

### Step A: 構成ロード

1. `ConfigLoader` で box 一覧を読み込む。
2. 各boxの `backend` を評価。
3. target -> box -> backend のマップを生成。

### Step B: controller 初期化

1. `qubecalib` controller を初期化（必要時のみ）。
2. `quelware-client` controller を初期化（必要時のみ）。
3. 各controllerの `connect` / `create_session` を行う。

### Step C: 実験実行

1. payload を backend ごとに分割。
2. legacy 側: 既存 sequencer 実行。
3. quel-3 側: directive 適用 -> trigger -> fetch_result。
4. 結果を共通フォーマットへ変換して集約。

### Step D: 終了処理

1. quelware session close。
2. legacy queue/connection cleanup。
3. 計測ログ（所要時間、失敗率）保存。

## 4. チェックリスト

### 4.1 設定

- [ ] すべての box に `type` が正しい。
- [ ] quel-3 box に `backend=quelware-client` が設定済み。
- [ ] quel-3 box に `controller_endpoint` が設定済み。

### 4.2 接続

- [ ] legacy controller connect 成功。
- [ ] quelware-client create_session 成功。
- [ ] 混在時に backend 切替マップが期待通り。

### 4.3 実行

- [ ] legacy payload が legacy controller のみで処理される。
- [ ] quel-3 payload が quelware-client controller のみで処理される。
- [ ] trigger 後に結果取得成功。
- [ ] 結果が共通フォーマットに正しく変換される。

### 4.4 異常時

- [ ] quel-3 側エラー時に legacy 側へ波及しない。
- [ ] controller 単位でリトライ/中止判断が可能。
- [ ] `legacy-only` で切戻し可能。

## 5. 推奨の段階導入

1. Phase 1: 読み取り専用（resource列挙、session open/close）
2. Phase 2: 単純 directive 実行（SetFrequency 等）
3. Phase 3: 小規模測定（1 instrument）
4. Phase 4: 既存実験フローへ統合

## 6. 記録テンプレート

## 実行情報

- 日時:
- 実行者:
- ブランチ/commit:
- 装置構成(legacy/quel-3):

## 接続結果

- legacy connect:
- quel-3 session:
- 失敗/再試行:

## 実行結果

- legacy 実行時間:
- quel-3 実行時間:
- 変換後データ整合:

## 判定

- Go/No-Go:
- 残課題:
- 次アクション:

## 7. 実装時の注意

- 既存コードへ直接 `if box.type == "quel3"` を散らさない。
  - controller selector + adapter 層に閉じ込める。
- 測定結果モデルは先に共通化し、backend固有型を漏らさない。
- 同期/トリガの責務境界（誰が時刻基準を持つか）を明文化する。
