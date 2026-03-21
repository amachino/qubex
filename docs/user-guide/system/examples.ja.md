# `system` サンプルワークフロー

このページでは、設定読み込み、system model の確認、実行前の設定チェックに
関する notebook 入口を紹介します。

## 推奨する出発点

- [ConfigLoader Example](../../examples/system/config_loader.ipynb): 設定ファイルを直接読み込み、生成された `ExperimentSystem` を確認します。
- [QuEL-1 skew 調整ワークフロー](../../examples/system/quel1_skew_adjustment.md): 同期した QuEL-1 box の現在 skew を確認し、`skew.yaml` を更新して再測定する流れをまとめています。
- [QuEL-3 Experiment Configure Check](../../examples/system/quel3_experiment_configure_check.ipynb): Measurement や Experiment の実行に進む前に、QuEL-3 の system 定義が意図どおり設定できるかを確認します。

## 関連ページ

- [システム設定](../getting-started/system-configuration.md)
- [`system` モジュール](index.md)
- [`backend` サンプルワークフロー](../backend/examples.md)
- [サンプル集全体](../../examples/index.md)
