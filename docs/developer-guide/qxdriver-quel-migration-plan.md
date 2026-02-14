# qxdriver-quel migration plan

## Goal

`packages/qube-calib` を、`qubex` から QuEL 装置 (quelware 0.10) を制御する専用ドライバーへ再構成する。

- リポジトリ名: `qxdriver-quel`
- Python パッケージ名: `qxdriver_quel`
- 互換条件: `qubex` が現在 `qubecalib` に依存している API は互換レイヤーとして維持
- 非互換条件: `qubex` が使っていない既存処理は削除

## Progress

- 2026-02-14: `qubex -> qubecalib` 互換 API の契約テストを追加。
- 2026-02-14: `qxdriver_quel` 名前空間を追加し、`qubecalib` への
  top-level / `qxdriver_quel.qubecalib` の互換 re-export を開始。
- 2026-02-14: `qxdriver_quel.clockmaster_compat` と
  `qxdriver_quel.instrument.quel.quel1.*` の import パス互換 shim を追加。

## 1. qubex が実際に使っている qubecalib API (互換対象)

以下は `src/qubex` を静的走査して得た「実使用 API」。

### 1.1 import パス互換

- `qubecalib`:
  - `QubeCalib`, `Sequencer`
  - `neopulse` モジュール (`src/qubex/backend/quel1/quel1_backend_controller.py:72`, `src/qubex/backend/quel1/quel1_backend_controller.py:1548`)
- `qubecalib.qubecalib`:
  - `Sequencer` (`src/qubex/backend/quel1/quel1_sequencer.py:5`)
  - `BoxPool`, `CaptureParamTools`, `Converter`, `WaveSequenceTools` (`src/qubex/backend/quel1/quel1_backend_controller.py:93`)
- `qubecalib.clockmaster_compat`:
  - `QuBEMasterClient`, `SequencerClient` (`src/qubex/backend/quel1/quel1_backend_controller.py:73`)
- `qubecalib.instrument.quel.quel1`:
  - `Quel1System` (`src/qubex/backend/quel1/quel1_backend_controller.py:77`)
- `qubecalib.instrument.quel.quel1.driver`:
  - `Action`, `AwgId`, `AwgSetting`, `NamedBox`, `RunitId`, `RunitSetting`, `TriggerSetting` (`src/qubex/backend/quel1/quel1_backend_controller.py:78`)
  - `single`, `multi` モジュール (`src/qubex/backend/quel1/execution/parallel_action_builder.py:410`)
- `qubecalib.instrument.quel.quel1.tool`:
  - `Skew` (`src/qubex/backend/quel1/quel1_backend_controller.py:87`)

### 1.2 QubeCalib オブジェクト互換

`Quel1BackendController` 経由で以下が呼ばれる。

- コンストラクタ
  - `QubeCalib()`
  - `QubeCalib(config_path)` (`src/qubex/backend/quel1/quel1_backend_controller.py:189`)
- メソッド
  - `define_clockmaster` (`src/qubex/backend/quel1/quel1_backend_controller.py:1380`)
  - `define_box` (`src/qubex/backend/quel1/quel1_backend_controller.py:1401`)
  - `define_port` (`src/qubex/backend/quel1/quel1_backend_controller.py:1426`)
  - `define_channel` (`src/qubex/backend/quel1/quel1_backend_controller.py:1454`)
  - `define_target` (`src/qubex/backend/quel1/quel1_backend_controller.py:1978`)
  - `modify_target_frequency` (`src/qubex/backend/quel1/quel1_backend_controller.py:1946`)
  - `show_command_queue` (`src/qubex/backend/quel1/quel1_backend_controller.py:1363`)
  - `clear_command_queue` (`src/qubex/backend/quel1/quel1_backend_controller.py:1367`)
  - `step_execute` (`src/qubex/backend/quel1/quel1_backend_controller.py:1652`)
- プロパティ
  - `executor` (なければ `_executor`) + `.add_command` (`src/qubex/backend/quel1/quel1_backend_controller.py:1357`)
  - `system_config_database` (`src/qubex/backend/quel1/quel1_backend_controller.py:242`)
  - `sysdb` (`src/qubex/backend/quel1/quel1_backend_controller.py:563`)

### 1.3 system_config_database / sysdb の互換

`qubex` は DB の public 属性と private 属性の両方にフォールバックしている。

- `asdict`, `asjson`
- `get_channels_by_target`, `get_channel`, `create_box`
- 属性 (public + private fallback):
  - `clockmaster_setting` / `_clockmaster_setting`
  - `box_settings` / `_box_settings`
  - `port_settings` / `_port_settings`
  - `target_settings` / `_target_settings`
  - `relation_channel_target` / `_relation_channel_target`
- `sysdb` 側:
  - `load_skew_yaml`
  - `assign_target_to_channel` (なければ `_relation_channel_target.append`)

### 1.4 Sequencer/実行系の互換

- `Sequencer` 生成時シグネチャ:
  - `gen_sampled_sequence`, `cap_sampled_sequence`, `resource_map`, `interval`, `sysdb`, `driver` (`src/qubex/backend/quel1/quel1_backend_controller.py:1512`)
- `Sequencer` 利用 API:
  - `execute(boxpool)`
  - `select_trigger(system, settings)`
  - `parse_capture_results(status, results, action, crmap)`
  - 属性 `interval`, `cap_sampled_sequence`, `gen_sampled_sequence`
- ツール:
  - `CaptureParamTools.create / enable_integration / enable_demodulation`
  - `WaveSequenceTools.create`
  - `Converter.multiplex`

### 1.5 neopulse 互換

`qubex` が構築に使う型:

- `DEFAULT_SAMPLING_PERIOD`
- `GenSampledSequence`, `GenSampledSubSequence`
- `CapSampledSequence`, `CapSampledSubSequence`, `CaptureSlots`

### 1.6 テスト互換 (qubex tests)

`src/qubex` 本体ではないが、現行テストが依存。

- `qubecalib.e7compat`: `CaptureParam`, `WaveSequence`
- `qubecalib.instrument.quel.quel1.driver.single`: `Action`, `AwgId`, `RunitId`

## 2. 新パッケージ設計 (qxdriver_quel)

`qubex` が必要とする互換 API だけを薄く提供し、コアは 0.10 ネイティブ設計に寄せる。

### 2.1 推奨ディレクトリ構成

```text
packages/qxdriver-quel/
  pyproject.toml
  src/qxdriver_quel/
    __init__.py
    facade.py                      # 新しい公開エントリ (driver-first)
    compat/
      __init__.py
      qubecalib.py                 # 旧 API 互換層 (QubeCalib, Sequencer, tools)
      e7compat.py                  # 必要最小限のみ
      clockmaster_compat.py
    config/
      system_db.py                 # SystemConfig DB abstraction
      models.py
    driver/
      quel1_system.py
      action/
        single.py
        multi.py
      types.py
    runtime/
      executor.py
      sequencer.py
      converter.py
    tool/
      skew.py
```

### 2.2 クラス責務

- `DriverSession` (新コア)
  - quelware 0.10 ドライバー操作、box/system lifecycle、action 実行
- `SystemConfigRepository`
  - 設定モデル管理。`qubex` 互換の最小 API を adapter で提供
- `CommandExecutor`
  - command queue / step execution
- `SequencerRuntime`
  - sampled sequence から direct action 設定への変換・実行
- `QubeCalibCompat` (`compat.qubecalib`)
  - 旧 `QubeCalib` シグネチャを受け取り、新コアへ委譲

## 3. 実装方針

### 3.1 互換は「明示的に凍結」

まず互換 API を契約として固定し、それ以外は実装しない。

- 互換 API テストを先に追加
- 互換対象外 API は削除
- private 属性 fallback は当面維持し、段階的に廃止

### 3.2 移行フェーズ

1. **Phase A: 互換面の固定**
   - `qubex` の実使用 API をテスト化
   - `qubecalib` 側の不要モジュール棚卸し
2. **Phase B: rename + package split**
   - `packages/qube-calib -> packages/qxdriver-quel`
   - import パスを `qxdriver_quel` に移行
   - `qubecalib` 名は compat shim として残す（当面）
3. **Phase C: 新コア移行**
   - `DriverSession`, `SystemConfigRepository`, `SequencerRuntime` を実装
   - compat 層を新コア委譲へ置換
4. **Phase D: 削除**
   - 互換対象外コード・テスト・ドキュメント削除
   - `qubex` 未使用 API の公開停止
5. **Phase E: hardening**
   - `uv run ruff check`
   - `uv run ruff format`
   - `uv run pyright`
   - `uv run pytest`

## 4. 直近の実施順 (次に着手する作業)

1. `qubex` 側に「互換 API 契約テスト」を追加し、現行実装でグリーン化
2. `packages/qube-calib` 内で、互換対象外モジュール一覧を作成
3. その一覧に基づいて `qxdriver_quel` 新構成へリネーム＋移設
4. compat shim を配置して `qubex` 側コード無変更で通る状態を作る

## 5. 成功条件

- `qubex` の backend テストが全てパス
- `qubex` から見た import path/動作が現行互換
- `qxdriver_quel` の公開 API が「QuEL ドライバー中心」に整理
- `qubex` 非依存の旧処理が削除済み
