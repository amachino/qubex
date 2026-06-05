# `backend` サンプルワークフロー

このページでは、backend 固有の実行経路と runtime 検証に関する notebook 入口を
紹介します。

backend レベルの tooling は backend family ごとに分かれています。QuEL-1 の
CW 出力は共通の `BackendController` 操作ではなく、`Quel1BackendController` の
デバッグ workflow として提供しています。

QuEL-1 の CW 出力 API の詳細は、[QuEL-1 連続波出力](continuous-wave.md) を参照してください。

## 推奨する出発点

- [PulseSchedule to QuEL-3 Sequencer Flow](../../examples/measurement/quel3_sequencer_builder_flow.ipynb): `PulseSchedule` が QuEL-3 の sequencer plan に変換される流れを確認します。
- [QuEL-3 Deploy Check](../../examples/system/quel3_deploy_check.ipynb): QuEL-3 環境で deploy と runtime 接続を検証します。
- [QuEL-1 連続波出力](continuous-wave.md): `Quel1BackendController` から CW 出力を直接開始・停止します。
- [QuEL-1 連続波 notebook](../../examples/backend/quel1_continuous_wave.ipynb): guard 付き notebook workflow で QuEL-1 CW デバッグを実行します。

## 関連ページ

- [低レベル API 概要](../low-level-apis/index.md)
- [`backend` モジュール](index.md)
- [`measurement` モジュール](../measurement/index.md)
- [`system` モジュール](../system/index.md)
- [サンプル集全体](../../examples/index.md)
