# Qubex 改善提案資料

作成日: 2025-08-22
対象ブランチ: develop

## 1. 概要

Qubex は量子実験（パルス制御・読み出し・キャリブレーション）を統合的に扱うフレームワーク。現状、実験ロジックを包含する巨大クラス (`Experiment` ~2000行) と計測 (`Measurement`) 周辺が中心となり、Jupyter Notebook による運用が前提化している。一方で、コードベースのモジュール化粒度・ドキュメント整備・自動化 (CI/CD)・テスト充足度に改善余地がある。本資料では、保守性 / 再現性 / 拡張性 / 信頼性 を向上させるための具体的改善案と 90 日ロードマップを提示する。

## 2. 評価観点

- アーキテクチャ：責務分離、拡張容易性
- コード品質：可読性、循環依存、巨大ファイル/関数
- テスト：カバレッジ、実行速度、決定性
- 型と静的解析：`py.typed` 提供済だが型厳密性のばらつき
- ドキュメント：ユーザ/開発者/運用/リファレンスの層別
- 依存管理：バージョンピン/最小互換性/optional extras
- パフォーマンス：計測/パルス生成/シミュレーションのホットスポット
- エラーハンドリング：例外階層/ユーザ向けメッセージ性
- ロギング・観測性：再現性確保、実験メタデータ
- 設定・キャリブレーションデータ管理：ライフサイクル・整合性検証
- ノートブック運用：結果のコード化、自動レポート化
- 自動化：CI、リリース、Lint、型、ドキュメントビルド
- セキュリティ/ライセンス：配布形態/依存ライセンス
- 開発プロセス：コントリビューションガイド、ブランチ戦略

## 3. 良い点 (Strengths)

- `pyproject.toml` によりモダンなビルド管理。
- `py.typed` 提供で型配布対応済。
- パルス/測定/実験の概念モデルが明確 (Pulse / PulseSchedule / Experiment)。
- 実験系クラスの高レベル API が整備され、Notebook からの利用性高。
- キャリブレーションノートとプロパティ永続化機構を保持。
- 例示ノートブックが豊富でユーザ学習コストを低減。

## 4. 改善優先度マトリクス (High / Medium / Low)

| 項目                                        | 優先度 | 期待効果                |
| ------------------------------------------- | ------ | ----------------------- |
| Experiment クラス分割 (責務分離)            | High   | 保守性/テスト容易性向上 |
| CI (lint/pytest/mypy) 導入                  | High   | 品質ゲート自動化        |
| 例外階層再整理 + ユーザ向けメッセージ       | High   | デバッグ性/UX 向上      |
| キャリブレーションデータ検証/バージョン管理 | High   | 再現性/安全性           |
| Rabi/CR 等解析結果の構造化 (DataModel)      | High   | 解析再利用性            |
| ドキュメント層別 (ユーザ/開発/内部設計)     | High   | オンボーディング短縮    |
| Notebook → スクリプト化パターン整備         | Medium | 再現性/CI 組込み        |
| 依存バージョン戦略 (最小上限指定)           | Medium | 互換性/更新容易性       |
| ロギング統合 (構造化 JSON + Rich)           | Medium | トレース/解析性         |
| パフォーマンス計測ベンチ (ベースライン)     | Medium | 最適化判断材料          |
| プラグイン/Backend 抽象レイヤ               | Medium | 新規 HW 対応容易性      |
| API 安定化ポリシー (SemVer)                 | Medium | 外部利用安心感          |
| pre-commit 導入                             | Low    | 開発体験向上            |
| ドキュメント自動生成 (mkdocs / sphinx)      | Low    | 継続的整備              |
| KPI/メトリクス (実験スループット等)         | Low    | 継続改善指標            |

## 5. 詳細改善提案

### 5.1 アーキテクチャ/モジュール構造

- `Experiment` (2000行) の責務を以下へ分割:
  - Session / Context (環境・接続管理)
  - CalibrationManager (キャリブレーション取得/更新/検証)
  - PulseLibrary / GateFactory (x90, zx90 生成ロジック)
  - Analysis (Rabi, RB, Tomography) 戦略クラス
  - Execution Engine (measure vs execute モード抽象)
- `measurement` と `experiment` の相互依存を Interface 化し循環回避。
- Cross Resonance / DRAG 等パルス生成パラメタを dataclass 化し純粋関数へ抽出。

### 5.2 コード品質/可読性

- 重複パターン: targets 引数の正規化 (str / list) → ユーティリティ関数化。
- 例: `targets` 正規化を decorator で DRY 化。
- 定数/デフォルト値: 一貫して `constants.py` 的モジュールへ集約。現状複数場所。
- 魔法値 (index % 4 in [0, 3]) に命名定数付与 (e.g. LOW_TO_HIGH_INDEX_SET)。

### 5.3 テスト

- 現在：pulse, measurement の一部ユニットテストのみ。experiment 高レベル検証不足。
- 追加:
  - キャリブレーション補正ロジック (correct_rabi_params 等) のプロパティテスト。
  - 疑似 backend (FakeDeviceController) による統合テスト。
  - Notebook の smoke テスト (nbconvert + 実行 1 セル短縮)。
  - カバレッジ収集 & 失敗閾値 (例: line 70%).

### 5.4 型ヒント/静的解析

- mypy を CI へ追加。`pyproject.toml` に mypy セクション定義。
- `TargetMap` の具体的 TypeVar / Protocol を強化。
- public API には Return Type 明示 (一部 `-> None` が欠如)。

### 5.5 ドキュメント

- 構成:
  - User Guide: セットアップ / 代表実験フロー / トラブルシュート
  - Developer Guide: モジュール設計図, 例外階層, 拡張方法
  - API Reference: 自動生成 (sphinx-autodoc or mkdocstrings)
  - Calibration Handbook: データフォーマット, 有効期限, バリデーション
- 各 Notebook 冒頭に "目的 / 前提 / 所要時間 / 期待成果" セクションを追加。

### 5.6 API 設計

- 安定 API と内部 API を明示 (`_` prefix / `__all__` / docs ラベル)。
- フルエントチェーンより副作用指向メソッドを整理 (e.g. correct_* 系は pure + return new data + save オプション)。
- エラー例: `get_pi_pulse` が未保存の場合例外→ fallback / 推奨ハンドリングパターンを docstring に記述。

### 5.7 依存関係/パッケージング

- 現在: `numpy ~=1.0` など非常に広い指定 (潜在的破壊的変更リスク)。
  - 推奨: 最低バージョンピン (e.g. `numpy >=1.24,<2.0`) + テスト通過上限調整。
- `httpx ~=0.0`, `jax ~=0.0`, `optax ~=0.0` は無効/誤設定に見える → 適切な最小版へ修正 or optional extra へ。
- オプション依存 (`backend`) を extras へ維持しつつ、追加: `docs`, `dev-all`。
- `pip == 24.0` 固定は不要かつ将来阻害 → 最小制約へ緩和。

### 5.8 パフォーマンス/効率

- ベースライン計測スクリプト追加 (pulse 合成、Rabi fitting、CR schedule 生成)。
- NumPy ベクトル化済箇所の JIT (jax / numba) 適用候補をホットスポット計測後に限定適用。
- 大規模測定データ: メモリコピー削減 (in-place 操作 / `__slots__` / 構造化配列)。

### 5.9 エラーハンドリング/例外階層

- `CalibrationMissingError` 以外も細分化: ConnectionError, ConfigSyncError, DataValidationError, HardwareTimeoutError など。
- 失敗時メッセージに "推奨次手順" を付記。
- 例外へ `context` (対象 qubit, file path) を属性として付与。

### 5.10 ロギング/観測性

- 現状: print と rich 混在。
- 提案: `logging` へ統一 + RichHandler (人間向け) + JSON (機械集計) のハンドラ二系統。
- 実験ごとに Run ID を発行し、ログ/結果/キャリブレーション紐付け。

### 5.11 設定/キャリブレーションデータ管理

- スキーマ (pydantic) で calib_note 構造検証—不正値( None 混入 ) をロード前に遮断。
- バージョンフィールド + マイグレーション (schema version up) スクリプト。
- 署名 (hash) + 変更履歴 (Git LFS or DVC) で再現性担保。

### 5.12 データモデル/シリアライズ

- `RabiParam` など dataclass + `to_dict()` / `from_dict()` 実装統一。
- 結果オブジェクトにメタデータ (日時, seed, commit hash, environment) を付与。
- 結果保存フォーマット: JSONLines + バイナリ (npz) 分離。

### 5.13 実験再現性/記録

- `ExperimentRecord` 生成時に environment snapshot: `pip freeze`, Git commit, config digest。
- ノートブック自動エクスポート (HTML + metadata) を post-run hook で保存。

### 5.14 Notebook 運用

- 冪等性: 乱数 seed, ハードウェア不要部分を `SimulationBackend` で切替。
- `scripts/convert_notebooks.sh` により `.ipynb` → `.py` (nbconvert) で差分レビュー容易化。
- Long-running 実験は Papermill でパラメトライズ & CI 実行可能に。

### 5.15 セキュリティ/ライセンス/リリース

- 依存ライセンスチェック (pip-licenses)。
- Release workflow (タグ → build → publish → GitHub Release Notes自動生成)。
- SBOM (cyclonedx) 生成で依存追跡。

### 5.16 自動化 (CI/CD)

- GitHub Actions ワークフロー例:
  - lint (ruff) + type (mypy) + test (pytest -n auto --fail-under=70)
  - build & publish (on tag)
  - docs build (mkdocs gh-pages)
  - nightly dependency update (dependabot/pip-tools)

### 5.17 開発プロセス

- `CONTRIBUTING.md` と `CODE_OF_CONDUCT.md` 追加。
- Conventional Commits + auto CHANGELOG (git-cliff)。
- PR テンプレ: 目的 / 変更点 / テスト / 逆互換性。

### 5.18 将来拡張

- ハードウェア抽象: DeviceController IF を Protocol 化しシミュレータ実装。
- 分散実験: マルチノード収集 (Ray / Dask) を `Executor` 抽象で差し替え可能に。
- 高レベル回路インポート (OpenQASM/QIR) → Pulse コンパイラ層。
- 最適化: GRAPE / CRAB / reinforcement tuning を plugin として登録。

## 6. 90 日ロードマップ (例)

| フェーズ | 期間    | 主要成果物                                                                           |
| -------- | ------- | ------------------------------------------------------------------------------------ |
| Phase 1  | 0-30日  | CI 導入, 依存バージョン整理, 例外階層設計, Experiment 分割設計書                     |
| Phase 2  | 31-60日 | Experiment リファクタ第一弾, pydantic スキーマ, ロギング統合, 基本ドキュメント       |
| Phase 3  | 61-90日 | 追加テスト/カバレッジ向上, Notebook 自動化, Release workflow, パフォーマンス計測基盤 |

## 7. 成功指標 (KPI 例)

- テストカバレッジ: 10% → 60% 以上
- CI 平均実行時間: < 8 分
- PR リードタイム: 平均 5 日 → 2 日
- 重大バグ (再現性/キャリブ破損) 発生率: 低減 (定義要)
- リリース頻度: 四半期 1 → 月次 1

## 8. 推奨ツール/設定例

- ruff: `ruff check --select ALL --ignore E501` (後で段階的縮小)
- mypy: strict optional, warn-redundant-casts, disallow-any-generics
- pre-commit: ruff, mypy (cache), nbstripout, trailing-whitespace
- docs: mkdocs-material + mkdocstrings (Python)
- dependency: pip-tools (compile minimal pins), dependabot

## 9. ディレクトリ再構成案 (案)

```
src/qubex/
  core/                # 基本型, 共通 util
  calibration/         # CalibrationNote, validation
  pulses/              # pulse/waveform/gate factory
  execution/           # 実行エンジン (hardware / simulation)
  experiment/          # 高レベルフロー (Rabi, RB, Tomography)
  analysis/            # fitting, visualization
  data/                # 保存/ロードモデル (schemas)
  logging/             # ログ & メタデータ
```

段階的移行で旧モジュールは deprecation docstring を付与。

## 10. リスクと緩和

| リスク                           | 緩和策                                           |
| -------------------------------- | ------------------------------------------------ |
| 大規模リファクタで短期生産性低下 | フェーズ分割 + 互換ラッパ維持                    |
| ハードウェア依存テストの不安定性 | シミュレーション backend 導入                    |
| キャリブファイル破損             | スキーマ検証 + バージョン/バックアップ自動化     |
| 依存更新による破壊               | 最小バージョン + CI マトリクス (Python 3.9-3.12) |

## 11. 次アクション (直近 2 週間)

1. 依存の異常指定 (`jax ~= 0.0` 等) 精査 & 修正 PR
2. GitHub Actions: lint + type + test ワークフロー追加
3. 例外階層 `qubex.exceptions` 追加 (既存移行は互換 alias)
4. `Experiment` 分割設計 (ADR: Architecture Decision Record 作成)
5. pydantic スキーマによる calib_note ローダ試作 & テスト
6. CONTRIBUTING.md 雛形追加

## 12. 付録: Architecture Decision Record (ADR) 推奨テンプレ

```
# ADR-XXXX: <タイトル>
Status: Proposed | Accepted | Superseded
Context: <背景 / 変化要因>
Decision: <採用案>
Alternatives: <検討した代替>
Consequences: 正 / 負
```

---
本資料は初期アセスメントに基づく。今後、計測パフォーマンス実データと運用フローの詳細をヒアリングし、更なる最適化機会を抽出予定。
