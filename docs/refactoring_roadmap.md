# Experiment リファクタリング計画とロードマップ

作成日: 2026年1月14日

このドキュメントでは、`Experiment` クラスの Mixin ベースから Composition ベースへの移行に伴う変更点と、今後の開発ロードマップについて記述する。

## フェーズ 1: Mixin から Composition への移行 (完了)

### 変更の概要
以前の `Experiment` クラスは、多数の Mixin クラス (`MeasurementMixin`, `CalibrationMixin` など) を多重継承することで機能を拡張していた。これにより、以下の問題が発生していた。

*   **巨大なクラス**: 多数のメソッドがフラットに露出し、クラスが肥大化。
*   **状態の散逸**: 各 Mixin が暗黙的に共有するインスタンス変数に依存しており、追跡が困難。

今回のリファクタリングにより、`Experiment` クラスは **Facade** パターン を採用し、内部で保持する `Context` や `Service` オブジェクトに処理を委譲する **Composition** (集約) 構成に変更された。

### 成果
*   `src/qubex/experiment/mixin/` および `src/qubex/experiment/protocol/` ディレクトリを廃止・削除した。
*   `Experiment` クラスは薄いラッパーとなり、主要な状態は `ExperimentContext` に集約された。

---

## フェーズ 2: Context の分割とサービスの名前空間化 (Next Step)

現在、`ExperimentContext` が非常に多くの責務（設定ロード、ターゲット解決、パルス定義など）を担っているため、これをより粒度の細かいサービスに分割する。

### 計画: Service の分割
`ExperimentContext` から特定のドメインロジックを切り出し、独立した Service クラスとする。

1.  **GateService (仮)**
    *   責務: デフォルトのゲート定数 (x90, y90 等) の定義、パルス波形の生成ロジック。
    *   現状: `Context` 内にハードコードされているパルス生成ロジックを移動。

2.  **TargetService (仮)**
    *   責務: `qubit_labels` など、制御対象の解決と管理。
    *   現状: `Context` 内で直接管理されているターゲット情報を移動。

### API の階層化と Deprecation
全てのメソッドを `Experiment` 直下にフラットに公開するのをやめ、機能カテゴリごとの名前空間（プロパティ）経由でのアクセスに移行する。

**例:**
```python
# Before (Current)
ex.x90(...)

# After (Planned)
ex.gate.x90(...)
```

既存のフラットなメソッド（`ex.x90` 等）は、互換性維持のために残さすが、`@deprecated` デコレータ等を付与し、将来的な削除を予告する。

---

## フェーズ 3: Command パターンによる測定の抽象化

定型的な測定手続きや、ユーザー定義の測定フローを統一的に扱うため、**Command** パターンを導入する。

### 概念: ExperimentTask
測定という行為を「実行可能なタスクオブジェクト」として定義する。

```python
@dataclass
class ExperimentTask:
    """全ての測定タスクの基底クラス (または Protocol)"""
    def execute(self, ctx: ExperimentContext) -> Result:
        ...
```

### 実行フローの変更
`Experiment` クラスのメソッド（あるいは Service のメソッド）は、具体的な処理を直接書くのではなく、`Task` を生成して `Executor` (Runner) に投げる形になる。

```python
# 概念コード
class Experiment:
    def run(self, task: ExperimentTask):
        # 共通の前処理、ログ、エラーハンドリング等はここで吸収
        return task.execute(self.ctx)

# ユーザーコード
task = RabiExperiment(...)
result = ex.run(task)
```

### ユーザー定義タスク
ユーザーは `ExperimentTask` を継承（または Protocol に準拠）したクラスを作成することで、ライブラリ内部に手を入れることなく、独自の測定プロセスを `ex.run()` のエコシステム上で実行できるようになる。

### Facade インタフェースの維持（重要）
Command パターン導入後も、Jupyter Notebook 等でのインタラクティブな利用における利便性を損なわないよう、`Experiment` クラスの Facade としての役割は維持する。頻繁に使用される測定（Rabi, Spectroscopy, T1など）については、ユーザーがいちいち Task クラスを import してインスタンス化しなくても済むよう、以下のようなショートカットメソッドを提供し続ける。

```python
# Notebook での利用イメージ
# 裏側では RabiExperiment が生成され run() されるが、
# ユーザーは引き続きメソッド呼び出しだけで完結できる。
result = ex.rabi_experiment(...)
```

これにより、構造のクリーンさと（内部実装の改善）と、ユーザー体験（手軽さ）の両立を図る。
