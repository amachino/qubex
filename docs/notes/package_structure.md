# src レイアウトにおけるパッケージ配置指針

最終更新: 2025-09-02

`src/` 配下には「配布 (wheel / sdist) 時にインポート可能であるべきコードのみ」を置く。メインパッケージが `qubex` の場合、原則としてすべては `qubex/` 以下に入れ、トップレベル直下に別パッケージ (ディレクトリ) を増やすのは次の明確な動機があるときに限定する。

## トップレベルに `qubex` 以外を置く主なケース

1. 複数パッケージ配布 (マルチパッケージ構成)

- 例: `qubex`, `qubex_labtools`, `qubex_plugins` を同一リポで同時にリリース。
- `pyproject.toml` の `packages` または自動発見で複数指定。

2. 名前空間パッケージ (PEP 420) の一部

- 組織名や共通 prefix を共有する複数パッケージ群を増やすとき。
- 例: `company.physics`, `company.control`。`__init__.py` を置かず implicit namespace に。

3. プラグイン / オプション機能の分離

- コア依存を軽く保ち、追加依存を extras_require で切り替えたい。
- 例: `qubex_sim`, `qubex_cloud`, `qubex_extras`。

4. 高速化ネイティブ拡張の分離

- C / Cython / Rust (maturin) 拡張をビルド要件ごと切り離し。
- 例: `qubex_core_ext` や `_qubex_native`。

5. 自動生成コード (API / プロトコル) の隔離

- gRPC / OpenAPI / protobuf / FlatBuffers など regen されるコードをまとめる。
- 例: `qubex_proto`。

6. データ専用パッケージ

- 大きくはない静的リソース (YAML, JSON, テンプレート) を importlib.resources で読む目的。
- 例: `qubex_data`。

7. 旧 API / 互換レイヤ保持

- 段階的廃止を進める旧実装を `qubex_legacy` などに分離し DeprecationWarning を集中管理。

8. CLI 集約

- 複数のエントリポイントを提供しコア依存方向を一方向にしたい。
- 例: `qubex_cli`。

9. 重い依存を持つシミュレータ / フェイクバックエンド

- コアインストールを軽量化したいケースで `qubex_simulator` など。

10. ベンダリング (例外的)

- 外部ライブラリをフォーク同梱。推奨: `qubex/_vendor/` にまとめトップ直下 `_vendor` は極力避ける。

11. 実験的機能の隔離

- 安定 API と明確に境界。例: `qubex_experimental`。

## まずは `qubex/` 配下に入れるべきもの

- 通常のサブドメイン: `analysis`, `backend`, `pulse`, `simulator`, etc.
- third-party ラッパ (改変しない/小規模) は `qubex.third_party.*` として配置。
- 内部利用のみのユーティリティ、定数、エラーモデル。

`qubex/third_party/` は他ライブラリ軽量ラッパや互換層をまとめる場所として十分。トップレベル新パッケージ化は「独立 import したい」「別バージョンサイクルにしたい」などの要件が生じてから検討。

## 判断基準 (チェックリスト)

- 独立バージョン / リリース管理が必要か
- 追加依存をコアから切り離したいか (optional extra)
- ネイティブビルド要件を分離したいか
- 自動生成再生成サイクルを人手編集コードと完全分離したいか
- 旧 API を見せつつ段階的廃止を管理したいか
- 利用者にトップレベル import 名前を追加で見せる必要が本当にあるか

YES が複数重なるならトップレベル分離を検討、そうでなければ `qubex/` に留める。

## アンチパターン / 避ける配置

- `tests/` を `src/` 直下に置く (インストール環境に混入)。
- 使い捨てスクリプトをパッケージに含める (`scripts/` や `tools/` へ)。
- 大容量生データを同梱 (別ストレージ / DVC / artifact)。
- トップレベルに多数の細かいパッケージを無計画に増殖。

## 推奨ワークフロー

1. 新機能はまず `qubex/` サブパッケージとして開始。
2. 分離要件が現れたら: 依存・公開 API・ビルド要件を整理した設計ノートを作成。
3. `pyproject.toml` で新パッケージを `packages` (もしくは自動発見設定) に追加し extras を設定。
4. ドキュメント: README と API リファレンスに新トップパッケージの目的と境界を明記。
5. CI: 分離後は最低 1 つの cross-import テスト (互換性) を追加。

## extras の例 (概念)

```toml
[project.optional-dependencies]
sim = ["qutip>=5"]
cli = ["rich", "typer"]
dev = ["pytest", "mypy", "ruff"]
all = ["qubex[sim,cli]"]
```

## 決定のまとめ (簡易フローチャート)

```
分離理由が明確? ── No ─→ qubex/ 内に配置
    │
    Yes
    │ (独立リリース or 重依存 or 生成物 or 旧API?)
    ▼
トップレベル新パッケージ作成 → extras 設定 / ドキュメント整備
```

## 運用メモ

- ランタイム import コストを考慮し、大型依存は遅延 import / optional extras。
- ベンダリング時は LICENSE 表記と元バージョンをファイルヘッダか `NOTICE` に明示。
- 実験的機能は `__all__` へ即座に公開しない (内部プレフィックス `_` or `experimental` セクション)。

---

このドキュメントは `src/` レイアウト運用ガイドラインの基準点として維持する。更新が必要になった場合は日付と変更点を追記すること。
