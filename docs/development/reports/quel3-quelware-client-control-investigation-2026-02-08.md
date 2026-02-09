# quel-3 (`quelware-client`) 併用制御の調査レポート（2026-02-08）

## 1. 調査目的

- 新規装置 `quel-3` を `quelware-client` 経由で制御する前提で、
  既存装置（`qubecalib`/`quel1` 経由）と同じ `qubex` から切替利用する方法を設計する。
- 「使用する装置に応じて自動/明示切替」できる実装方針を整理する。

## 2. 事実確認（現状）

### 2.1 qubex 側

- backend は実質 `Quel1BackendController` 固定で、`qubecalib` 依存が強い。
- box 定義は `BoxType` が `quel1*` / `qube*` 中心で、`quel-3` 型が未定義。
- config 読み込み (`box.yaml`) から backend 種別を分岐する仕組みはない。

### 2.2 quelware-client 側

- クライアントは gRPC ベースで `create_quelware_client(endpoint, port)` から開始。
- 主要操作は下記。
  1. `list_resource_infos()` で利用可能 resource 列挙
  2. `create_session(resource_ids)` でセッション確保
  3. `deploy_instruments(port_id, definitions)` で instrument 構成
  4. `create_instrument_driver_fixed_timeline(...)`
  5. `driver.apply([...directives...])`
  6. `session.trigger()`
  7. `driver.fetch_result()`
- `quelware-client` は Python `>=3.12` 要件。

## 3. 差分整理（旧来 vs quel-3）

- 旧来 (`qubecalib`) は「box/port/ch/runit 直接構成」中心。
- `quelware-client` は「resource/session/instrument + directive」中心。
- API モデルが異なるため、`Quel1BackendController` にif文を増やすより、
  backend adapter 分離が必須。

## 4. 推奨アーキテクチャ

### 4.1 backend interface を導入

`qubex.backend.controllers` に共通 interface を作る。

想定メソッド（最小）

- `connect(targets_or_boxes, **options)`
- `link_status(...)`
- `sync_clocks(...)`
- `apply_port_channel_config(...)`
- `execute(payload)`
- `fetch_box_or_resource_snapshot(...)`

実装

- `Quel1LegacyBackendController`（既存 `qubecalib`）
- `Quel3ClientBackendController`（新規 `quelware-client`）

### 4.2 実行経路の分離

- `MeasurementBackendAdapter` も装置種別で実装を分ける。
  - `Quel1MeasurementBackendAdapter`
  - `Quel3MeasurementBackendAdapter`
- sequencer/payload 生成は共通部と実装依存部に分割する。

## 5. 切替方法（設定駆動）

## 5.1 box.yaml 拡張案

既存 `box.yaml` に `backend` と `controller_endpoint` を追加する。

```yaml
Q3A:
  name: "Quel-3 A"
  type: "quel3"
  address: "10.20.0.10"
  adapter: null
  backend: "quelware-client"
  controller_endpoint: "10.20.0.2:50051"
```

既存装置は従来通り。

```yaml
Q2A:
  name: "Quel1 A"
  type: "quel1-a"
  address: "10.3.0.11"
  adapter: "adapter-name"
  backend: "qubecalib"
```

## 5.2 選択ルール

1. `backend` 明示があれば優先。
2. 無ければ `type` から推定（`quel3` -> `quelware-client`、それ以外 -> `qubecalib`）。
3. 同一実験に複数backendが混在する場合は、対象target単位で controller を振り分ける。

## 6. 実装ステップ（推奨順）

1. `BoxType` に `QUEL3` 追加。
2. `ConfigLoader` で `backend` / `controller_endpoint` をパース。
3. `SystemManager` に controller レジストリを導入（`backend_kind -> controller`）。
4. 既存 `Quel1BackendController` を `Quel1LegacyBackendController` として移設。
5. `Quel3ClientBackendController` 新規実装。
   - `create_quelware_client`
   - resource/instrument mapping
   - session lifecycle
   - trigger/fetch result
6. `MeasurementBackendManager` で controller 選択ロジック追加。
7. 旧実装と新実装のA/B切替フラグを残してベンチ。

## 7. リスクと対策

- Python要件差（3.10 vs 3.12）:
  - 対策: `quelware-client` を optional extra に分離し、専用実行環境を用意。
- data model差（box-centric vs instrument-centric）:
  - 対策: adapter 層で payload 変換責務を局所化。
- 同期/trigger仕様差:
  - 対策: 実機回帰項目に「時刻同期・トリガ整合」を必須化。

## 8. 冗長処理の可能性（現時点）

- 旧来の `connect` 後に `pull` で再取得している情報の一部は、
  `quelware-client` 側では resource cache と二重管理になる可能性がある。
- `box_config` 前提の downstream 処理は `quel3` では意味が異なるため、
  snapshot抽象（box/resource共通）に寄せないと重複変換が増える。

## 9. 参考ソース

- quelware-client repository:
  - https://github.com/quel-inc/quelware-client
- client entrypoint:
  - https://github.com/quel-inc/quelware-client/blob/main/quelware-client/src/quelware_client/client/_grpc.py
- session API:
  - https://github.com/quel-inc/quelware-client/blob/main/quelware-client/src/quelware_client/core/_session.py
- instrument usage example:
  - https://github.com/quel-inc/quelware-client/blob/main/quelware-client/examples/use_instrument.py
- deploy example:
  - https://github.com/quel-inc/quelware-client/blob/main/quelware-client/examples/deploy_instrument.py
- qubex current controller:
  - /Users/amachino/qubex/src/qubex/backend/quel1/quel1_backend_controller.py
