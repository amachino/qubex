# qubecalib `quelware_0.10.x` 移行調査レポート（2026-02-08）

## 1. 目的

- `qubex` の現行利用方法のまま `qubecalib` を `quelware_0.10.x` 系へ移行できるかを調査する。
- 互換性が崩れる箇所と、移行のための修正方針を整理する。
- `3.1.16beta2` と `quelware_0.10.x` の差分を分かりやすくまとめる。

## 2. 比較対象

- 現行 `qubex` backend 依存: `qubecalib @ ...@3.1.16beta1`（`/Users/amachino/qubex/pyproject.toml`）
- 比較A: `qubecalib` tag `3.1.16beta2`（commit `8a5dbb7d`）
- 比較B: `qubecalib` branch `feat/quelware_0.10.x`（commit `d8610122`）

## 3. 実施内容

- `qubex` 側の `qubecalib` 依存 API 呼び出し箇所を抽出。
- `qubecalib` の比較2系統（`3.1.16beta2` / `feat/quelware_0.10.x`）で API 有無を静的比較。
- 実行スモーク:
  - ホスト環境で `3.1.16beta2` を導入して import/属性確認。
  - devcontainer 相当コンテナ（`mcr.microsoft.com/devcontainers/base:bookworm`）で `feat/quelware_0.10.x` を導入して import/属性確認。

## 4. 結論（要点）

- `3.1.16beta2` は現行 `qubex` 利用と**概ね互換**。
- `feat/quelware_0.10.x` は現行 `qubex` と**非互換（そのままでは移行不可）**。
- 主因は以下の2点。
  1. `qubecalib` トップレベルの `QubeCalib`/`Sequencer` 再公開削除。
  2. `QubeCalib` から `create_box` / `create_boxpool` / `read_clock` / `resync` が削除。

## 5. 互換性崩れの詳細（qubex 側影響）

### 5.1 import 互換

- 影響箇所:
  - `/Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py:64`
  - `/Users/amachino/qubex/src/qubex/experiment/experiment_tool.py:18`
- 現状コードは `from qubecalib import QubeCalib, Sequencer` 前提。
- `quelware_0.10.x` では `qubecalib/__init__.py` から再公開されないため ImportError。

### 5.2 メソッド互換

- 影響箇所（代表）:
  - `/Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py:386`
  - `/Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py:402`
  - `/Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py:574`
  - `/Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py:613`
- 呼び出し先の `QubeCalib` メソッド（`create_box` / `create_boxpool` / `read_clock` / `resync`）が `quelware_0.10.x` 側で消失。

### 5.3 `quel_clock_master` 依存の構造変化

- `quelware v0.8.14` には `quel_clock_master` サブパッケージあり。
- `quelware v0.10.5` ではモノレポ構成が変わり `quel_clock_master` サブディレクトリは存在しない。
- `qubex` には直接 import が残る:
  - `/Users/amachino/qubex/src/qubex/experiment/experiment_tool.py:17`
- このため、環境構築方法次第で追加の import 互換対応が必要。

## 6. `3.1.16beta2` vs `quelware_0.10.x` 差分サマリ

| 観点 | `3.1.16beta2` | `quelware_0.10.x` | 影響 |
| --- | --- | --- | --- |
| `qubecalib.__init__` | `QubeCalib`, `Sequencer`, `neopulse` を公開 | 公開削除（コメントアウト） | 既存 import が壊れる |
| `QubeCalib.create_box` | あり | なし | box取得系 API が壊れる |
| `QubeCalib.create_boxpool` | あり | なし | `connect()` が壊れる |
| `QubeCalib.read_clock` | あり | なし | clock確認 API が壊れる |
| `QubeCalib.resync` | あり | なし | clock再同期 API が壊れる |
| `QubeCalib.define_target` ほか | あり | あり | 一部 API は継続 |
| `sysdb.create_quel1system` | あり | あり（実装更新） | 直呼びなら活用可能 |
| `quel_ic_config` 依存 | git `v0.8.14` | `quel_ic_config==0.10.5` | 依存形態が変化 |
| 追加依存 | なし | `PyGObject` 追加 | ビルド要件増加（cairo 等） |

## 7. スモーク検証結果

### 7.1 `3.1.16beta2`

- `qubecalib` import: 成功
- `from qubecalib import QubeCalib`: 成功
- `QubeCalib` 重要メソッド（`create_box`, `create_boxpool`, `read_clock`, `resync`）: 存在確認
- `qubex.backend.quel1.quel1_backend_controller` import: 成功

### 7.2 `quelware_0.10.x`（devcontainer 相当環境）

- `qubecalib` / `qubecalib.qubecalib` import: 成功
- `from qubecalib import QubeCalib`: **失敗**
- `QubeCalib` の `create_box`, `create_boxpool`, `read_clock`, `resync`: **未定義**

## 8. 移行するための修正方針

### 方針A（推奨）: qubecalib 側で後方互換レイヤを戻す

- `qubecalib.__init__` で `QubeCalib`/`Sequencer` 再公開を復活。
- `QubeCalib` に以下の薄い互換メソッドを復活（内部は新 API へ委譲）。
  - `create_box`
  - `create_boxpool`
  - `read_clock`
  - `resync`
- 利点: `qubex` 変更を最小化し、既存ユーザーコードの破壊を抑制できる。

### 方針B: qubex 側で両対応アダプタを実装

- import fallback:
  - まず `from qubecalib import ...` を試し、失敗時 `from qubecalib.qubecalib import ...` にフォールバック。
- `Quel1BackendController` 内部に互換ラッパを追加。
  - `self.qubecalib.create_box` が無い場合は `self.qubecalib.sysdb.create_box` を利用。
  - `create_boxpool` は `BoxPool` を直接組み立てる分岐を実装。
  - `read_clock` / `resync` は `Quel1Box` / clockmaster API を直接使う代替を実装（要実機検証）。
- `experiment_tool.py` の `quel_clock_master` 直接依存は `quel_ic_config` 側 API へ置換（または条件付き import）。

## 9. 推奨移行手順

1. 先に `qubecalib` 側へ方針Aの互換復元を提案/適用。
2. `qubex` では方針Bの最小フォールバック（import fallback）だけ先行導入。
3. 実機を使った以下の回帰確認を実施。
   - box 接続/再接続
   - clock 読み出し/同期
   - sequencer 実行（capture 含む）
4. 問題なければ `qubex` の依存を段階的に `quelware_0.10.x` 系へ更新。

## 10. 補足リスク

- `quelware_0.10.x` は `qubecalib` 側バージョン表示が `3.2.0alpha` のため、安定運用前提なら API 固定化までリスクが残る。
- `PyGObject`/`pycairo` などビルド依存が増えており、環境差分で導入失敗が起こりやすい。
