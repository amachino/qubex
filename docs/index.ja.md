# Qubex

Qubex は、[QuEL](https://quel-inc.com/) の制御装置をバックエンドに、パルスレベルの実験ワークフローを一元化するフレームワークです。デバイス設定から任意のパルスシーケンス実行、評価、キャリブレーション、ベンチマーク、パルスレベルシミュレータによるオフライン反復までをスムーズに支えます。

多くの利用者にとって、推奨される利用の始め方は `Experiment` です。

> [!NOTE]
> このサイトは英語を既定言語にしています。
> まだ日本語化されていないページでは、英語版が表示されます。

## 主な特長

- **エンドツーエンドのワークフロー**: セットアップから実行、解析までを一貫した流れで進められます。
- **バックエンド統合セットアップ**: 設定ファイルでデバイスを定義し、QuEL の制御装置を用いた制御リソースの準備を Qubex に任せられます。
- **パルスレベル制御**: 回路レベルの制約を超えた柔軟なパルス系列を実行できます。
- **標準化された実験ルーチン**: 評価、較正、ベンチマークのワークフローを共通化できます。
- **パルスレベルシミュレータ**: 実験と同じパルス系列をハミルトニアンレベルでシミュレーションできます。

## まず読む

- [インストール](user-guide/getting-started/installation.md)
- [システム設定](user-guide/getting-started/system-configuration.md)
- [始め方を選ぶ](user-guide/getting-started/choose-where-to-start.md)
- [PulseSchedule の組み方](user-guide/pulse-sequences/index.md)

## 推奨される導線

- [Experiment ガイド](user-guide/experiment/index.md): 実機実験の多くで推奨されるユーザー向けワークフローから始めます。
- [Simulator ガイド](user-guide/simulator/index.md): 実機を使わずにパルスレベルのハミルトニアンダイナミクスを扱います。

## 低レベル API

- [概要](user-guide/low-level-apis/index.md): measurement 側の抽象化を主語にしたい場合の入口です。
- [Measurement API 概要](user-guide/measurement/index.md): session、schedule、capture/readout、sweep、backend integration を直接扱います。

## サンプルと API

- [サンプル集](examples/index.md)
- [API リファレンス](https://amachino.github.io/qubex/api-reference/qubex/)

## Qubex にコントリビュートする

- [Contributing](CONTRIBUTING.md)
- [開発ガイド](developer-guide/index.md)
