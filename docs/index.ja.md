# Qubex

Qubex は、[QuEL](https://quel-inc.com/) の制御装置をバックエンドに、パルスレベルの実験ワークフローを一元化する量子ビット制御実験フレームワークです。デバイス設定から任意のパルスシーケンス実行、量子デバイスの特性評価、量子ゲートのキャリブレーション、ベンチマーク、およびパルスレベルシミュレータによるシミュレーション実験までをスムーズに支えます。

> [!NOTE]
> このサイトは英語を既定言語にしています。
> まだ日本語化されていないページでは、英語版が表示されます。

## 主な特長

- **エンドツーエンドのワークフロー**: 実験系のセットアップからパルスシーケンスの実行、結果の解析までを一貫した流れで進められます。
- **バックエンド統合セットアップ**: 設定ファイルを用意するだけで、複数の QuEL 制御装置モデルを組み合わせた量子ビット制御実験のセットアップを Qubex に任せられます。
- **パルスレベル制御**: 回路レベルの制約を超えた柔軟なパルスシーケンスを実行できます。
- **標準化された実験ルーチン**: 量子デバイスの特性評価、較正、ベンチマークのワークフローを共通化できます。
- **パルスレベルシミュレータ**: 実験と同じパルスシーケンスをハミルトニアンレベルでシミュレーションできます。

## まず読む

- [インストール](user-guide/getting-started/installation.md)
- [始め方を選ぶ](user-guide/getting-started/choose-where-to-start.md)
- [システム設定](user-guide/getting-started/system-configuration.md)
- [パルスシーケンスの組み方](user-guide/pulse-sequences/index.md)

## 推奨される導線

- [`Experiment` クラス](user-guide/experiment/index.md): 実機実験の多くで推奨されるユーザー向けワークフローから始めます。
- [QuantumSimulator](user-guide/simulator/index.md): 実機を使わずにパルスレベルのハミルトニアンダイナミクスを扱います。

## 低レベル API

- [概要](user-guide/low-level-apis/index.md): session や schedule など、measurement 側の概念を中心に見たい場合の入口です。
- [Measurement API 概要](user-guide/measurement/index.md): session、schedule、capture/readout、sweep、backend integration を直接扱います。

## サンプルと API

- [サンプル集](examples/index.md)
- [API リファレンス](api-reference/qubex/index.md)

## Qubex にコントリビュートする

- [Contributing](CONTRIBUTING.md)
- [開発ガイド](developer-guide/index.md)
