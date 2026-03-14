# `backend` サンプルワークフロー

このページでは、backend 固有の実行経路と runtime 検証に関する notebook 入口を
紹介します。

ここで扱う notebook は、backend レベルの tooling が backend family ごとに
分かれるため、現時点では QuEL-3 寄りです。

## 推奨する出発点

- [PulseSchedule to QuEL-3 Sequencer Flow](../../examples/measurement/quel3_sequencer_builder_flow.ipynb): `PulseSchedule` が QuEL-3 の sequencer plan に変換される流れを確認します。
- [QuEL-3 Deploy Check](../../examples/system/quel3_deploy_check.ipynb): QuEL-3 環境で deploy と runtime 接続を検証します。

## 関連ページ

- [低レベル API 概要](../low-level-apis/index.md)
- [`backend` モジュール](index.md)
- [`measurement` モジュール](../measurement/index.md)
- [`system` モジュール](../system/index.md)
- [サンプル集全体](../../examples/index.md)
