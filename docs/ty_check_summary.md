# ty check 出力サマリ（種別別）

対象コマンド: `uv run ty check --output-format=github`

本書は現行の `ty` 出力を、種別ごとに整理し、解説と対応方針をまとめたものです。詳細な発生箇所は実行ログを参照してください。

## 全体方針（簡潔）
- **警告（warning）**は「将来的なバグ源」または「型情報の欠落」を示すため、原則として**新規追加を禁止**、既存は段階的に解消。
- **エラー（error）**は型不整合による誤用が確定的なため、**優先的に修正**。
- 既存の互換性やAPI公開を尊重しつつ、**型注釈・TypedDict・プロトコル・`cast`**で段階的に精度を上げる。

---

## Warnings

### possibly-missing-attribute
**概要**: 参照先オブジェクトの型に、その属性・メソッドが存在しない可能性がある。

**主な発生箇所（概略）**:
- `analysis/iq_plotter.py`（Plotly `FigureWidget` の `add_scatter` / `add_scatterpolar`）
- `experiment/services/optimization_service.py`（`cma.CMAEvolutionStrategy`）
- `tests/pulse/*`（`qubex` への re-export が見えない）

**方針**:
- 外部ライブラリは **stub/typing** が不足しがちなので、**`Protocol` か `cast`** で明示。
- `qubex` の re-export を意図しているなら、**`__init__.py` に明示的に追加**。
- Optional / Union 型は **`isinstance` で絞り込み**。

---

### unused-ignore-comment
**概要**: `# type: ignore` が不要で、実際にエラーを抑制していない。

**主な発生箇所（概略）**:
- `analysis/iq_plotter.py`, `backend/config_loader.py`, `diagnostics/chip_inspector.py`
- `experiment/services/*`, `measurement/*`, `simulator/*`

**方針**:
- **不要な ignore を削除**し、抑制が必要なら **対象エラーコードを限定**。
- 例: `# type: ignore[<code>]` に限定して意図を明確化。

---

### invalid-ignore-comment
**概要**: `type: ignore` の書式エラー（空白不足など）。

**方針**:
- 書式を修正（`# type: ignore`）か、不要なら削除。

---

### deprecated
**概要**: 既に非推奨指定された関数・クラスの使用。

**主な発生箇所（概略）**:
- `experiment/experiment.py`（`register_custom_target`, `save_defaults` など）
- `simulator/deprecated/*`

**方針**:
- **新規利用は停止**。既存は **移行ガイドを併記**しながら段階的に置換。
- 代替APIが明記されている場合は **置換を優先**。

---

## Errors

### invalid-argument-type
**概要**: 関数・メソッド引数の型が合っていない。

**主な発生箇所（概略）**:
- `analysis/state_tomography.py` / `measurement/measurement.py` / `measurement_result.py`（`reduce` の型不一致）
- `experiment/services/*` / `experiment_result.py`（Literal 制約、TypedDict への dict 渡し）
- `backend/experiment_system.py`（`GenChannel | CapChannel` の混在）

**方針**:
- **型注釈の絞り込み**、`Literal` の受け渡し修正。
- `Union` の場合は **分岐で狭める**か **専用メソッドに分離**。
- `dict` で渡している箇所は **`TypedDict` を構築**。

---

### invalid-assignment
**概要**: 属性・フィールドへの代入の型が不一致。

**主な発生箇所（概略）**:
- `backend/control_system.py`（`nwait`, `ndelay`）
- `experiment/services/calibration_service.py` / `measurement_service.py`（`CalibrationNote` への代入）

**方針**:
- **型付きプロパティの仕様に合わせる**。
- 外部入力は **一度ローカル変数で正規化**してから代入。

---

### invalid-method-override
**概要**: 継承先メソッドのシグネチャが基底と互換でない。

**主な発生箇所（概略）**:
- `pulse/library/*`（`Pulse.func` の override）
- `measurement/state_classifier_*`（`StateClassifier.fit`）
- `backend/sequencer_mod.py`

**方針**:
- **引数・戻り値の型を基底に合わせる**。
- 互換性が必要なら **`@overload`** で表現、または **基底クラスの型定義を見直す**。

---

### invalid-return-type
**概要**: 戻り値の型が注釈と一致しない。

**主な発生箇所（概略）**:
- `backend/sequencer_mod.py`, `pulse/waveform.py`, `simulator/quantum_system.py`

**方針**:
- 実返却値に合わせて **注釈を修正**、または **返却値をキャスト/変換**。

---

### unresolved-import
**概要**: import 先が解決できない。

**主な発生箇所（概略）**:
- `experiment/__init__.py`, `experiment/experiment.py`（`.experiment_task`）

**方針**:
- 実体が存在するなら **パス修正**。
- 廃止済みなら **import 削除**。

---

### unresolved-attribute
**概要**: オブジェクトに存在しない属性を参照。

**主な発生箇所（概略）**:
- `experiment/services/measurement_service.py`（`object.repeated`）

**方針**:
- 対象を **具体型に絞り込む**。
- 属性名の誤りなら **正しい名前に修正**。

---

### no-matching-overload
**概要**: 関数オーバーロードの候補に合致しない。

**主な発生箇所（概略）**:
- `analysis/visualization.py`（`real`, `imag`）
- `simulator/deprecated/system.py`（`tensor`）

**方針**:
- 受け渡し型の**明示キャスト**。
- 期待型に変換してから呼び出す。

---

### unsupported-operator
**概要**: オペレーターが対象型で定義されていない。

**主な発生箇所（概略）**:
- `experiment/services/optimization_service.py`
- `simulator/deprecated/*`

**方針**:
- **演算対象を型変換**（数値・配列化）してから演算。
- 必要なら **演算メソッドをラップ**。

---

### not-subscriptable / not-iterable
**概要**: 添字アクセス/反復ができない型に対する操作。

**主な発生箇所（概略）**:
- `experiment/services/characterization_service.py`
- `experiment/services/optimization_service.py`

**方針**:
- **`None` チェック**、**型ガード**を追加。
- 期待する型（`Mapping` / `Sequence`）へ変換。

---

### missing-typed-dict-key
**概要**: `TypedDict` に必要キーが不足。

**主な発生箇所（概略）**:
- `experiment/services/calibration_service.py`
- `tests/experiment/test_calibration_note.py`

**方針**:
- **必要キーを全て埋める**。
- 省略可能であれば **`NotRequired`** を検討。

---

## 進め方の提案（短期〜中期）
1. **構造的に影響が大きいもの**（`invalid-method-override` / `unresolved-import` / `invalid-argument-type`）から優先修正。
2. **`TypedDict` の整備**で `invalid-argument-type` / `missing-typed-dict-key` を縮減。
3. **unused ignore の掃除**で警告ノイズを減らし、次の修正を容易にする。

---

## 参考
- `ty` 出力は CI で `--output-format=github` により収集。
- 具体的な修正計画はモジュール単位で別ドキュメント化推奨。
