# Qubex

Qubex は、[QuEL](https://quel-inc.com/) の制御装置をバックエンドに、パルスレベルの実験ワークフローを一元化する量子ビット制御実験フレームワークです。デバイス設定から任意のパルスシーケンス実行、量子デバイスの特性評価、量子ゲートのキャリブレーション、ベンチマーク、およびパルスレベルシミュレータによるシミュレーション実験までをスムーズに支えます。

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

- [`Experiment` クラス](user-guide/experiment/index.md): 実機実験の多くで推奨される高レベルな入口です。
- [`QuantumSimulator` クラス](user-guide/simulator/index.md): 実機を使わずにパルスレベルのハミルトニアンダイナミクスを扱う入口です。

## 低レベル API

- [概要](user-guide/low-level-apis/index.md): `measurement`、`system`、`backend` の役割分担を整理した入口です。
- [`measurement` モジュール](user-guide/measurement/index.md): `MeasurementSchedule`、キャプチャ/読み出し、sweep、`measurement` の実行フローを直接扱います。
- [`system` モジュール](user-guide/system/index.md): 設定読み込み、`ExperimentSystem`、実行時状態の同期を扱います。
- [`backend` モジュール](user-guide/backend/index.md): backend controller、execution request、QuEL 固有実装を扱います。

## サンプルと API

- [サンプル集](examples/index.md)
- [リリースノート](release-notes/index.md)
- [API リファレンス](api-reference/qubex/index.md)

## Qubex にコントリビュートする

- [Contributing](CONTRIBUTING.md)
- [開発ガイド](developer-guide/index.md)
