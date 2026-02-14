# qxdriver-quel shim reduction inventory

## Purpose

`packages/qube-calib` を最終的に `qxdriver_quel` 中心へ整理するために、
`qubecalib` 側で残すべき互換面と削除候補を明確化する。

## Evidence scope

- `src/qubex` と `tests/backend` に対する参照走査
- `packages/qube-calib/src/qxdriver_quel` が参照する `qubecalib` 走査

## Keep: required for qubex runtime

以下は `qubex` 実行系の互換で必要。

- `packages/qube-calib/src/qubecalib/__init__.py`
- `packages/qube-calib/src/qubecalib/qubecalib.py`
- `packages/qube-calib/src/qubecalib/facade.py`
- `packages/qube-calib/src/qubecalib/clockmaster_compat.py`
- `packages/qube-calib/src/qubecalib/neopulse.py`
- `packages/qube-calib/src/qubecalib/e7compat.py`
- `packages/qube-calib/src/qubecalib/e7utils.py`
- `packages/qube-calib/src/qubecalib/sysconfdb.py`
- `packages/qube-calib/src/qubecalib/resource_map.py`
- `packages/qube-calib/src/qubecalib/tree.py`
- `packages/qube-calib/src/qubecalib/runtime/*`
- `packages/qube-calib/src/qubecalib/instrument/quel/quel1/*`

## Keep: required for qubex test compatibility

以下は `tests/backend` の互換テストが直接 import。

- `packages/qube-calib/src/qubecalib/e7compat.py`
- `packages/qube-calib/src/qubecalib/instrument/quel/quel1/driver/single.py`

## Candidate remove: not referenced by qubex and qxdriver_quel

現時点の走査で `qubex` と `qxdriver_quel` のどちらからも参照なし。

- `packages/qube-calib/src/qubecalib/task/interface.py`
- `packages/qube-calib/src/qubecalib/utils/__init__.py`
- `packages/qube-calib/src/qubecalib/utils/pca.py`
- `packages/qube-calib/src/qubecalib/units.py`

関連テスト (同時削除候補):

- `packages/qube-calib/tests/unit/units/test_unit.py`

## Suggested execution order

1. 上記 candidate remove を 1コミットで削除
2. `packages/qube-calib/tests/unit` のうち削除対象に紐づくテストを同時削除
3. `uv run pytest tests/backend -q` を通して `qubex` 互換を確認
4. 問題なければ、`qubecalib` 側を thin shim 方針へ段階移行

## Notes

- このインベントリは `qubex` 観点の最適化。外部利用者向け公開 API は別途評価が必要。
- 旧 `qubecalib` のディレクトリ構成維持は必須ではないが、
  `qubex` が使う互換シンボルは維持する。
