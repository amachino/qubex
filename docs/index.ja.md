# Qubex

Qubex は、異種デバイスの設定、任意のパルス系列の構築と実行、標準的な評価・較正・ベンチマーク手順、そしてパルスレベルシミュレータによるオフライン反復を、ひとつのフレームワークにまとめます。

推奨される利用の始め方は `Experiment` です。
より高度な制御が必要な場合は、下位レベルの `Measurement` API を利用できます。

> [!NOTE]
> このサイトは英語を既定言語にしています。
> まだ日本語化されていないページでは、英語版が表示されます。

## 主な特長

- **エンドツーエンドのワークフロー**: セットアップから実行、解析までを一貫した流れで進められます。
- **複数ハードウェアの自動セットアップ**: 設定ファイルでデバイスを定義し、混在した制御スタックを Qubex に自動構成させられます。
- **パルスレベル制御**: 回路レベルの制約を超えた柔軟なパルス系列を実行できます。
- **標準化された実験ルーチン**: 評価、較正、ベンチマークのワークフローを共通化できます。
- **パルスレベルシミュレータ**: 実験と同じパルス系列をハミルトニアンレベルでシミュレーションできます。

## まず読む

- [インストール](user-guide/getting-started/installation.md)
- [システム設定](user-guide/getting-started/system-configuration.md)
- [入口を選ぶ](user-guide/getting-started/choose-your-entry-point.md)

## ワークフローを選ぶ

- [Experiment ガイド](user-guide/experiment/index.md): 実機を使う高レベルな実験ワークフローから始めます。
- [Measurement ガイド](user-guide/measurement/index.md): 測定セッション、カスタムスケジュール、低レベル readout フローを直接扱います。
- [Simulator ガイド](user-guide/simulator/index.md): 実機を使わずにパルスレベルのハミルトニアンダイナミクスを扱います。

## サンプルと API

- [サンプル集](examples/index.md)
- [API リファレンス](https://amachino.github.io/qubex/api-reference/qubex/)

## Qubex にコントリビュートする

- [Contributing](CONTRIBUTING.md)
- [開発ガイド](developer-guide/index.md)
