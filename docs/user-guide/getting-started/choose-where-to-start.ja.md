# 始め方を選ぶ

Qubex の使い始め方は、まず環境を用意し、その後で目的に合う入口へ進む流れです。
このページでは、実際の進み方に合わせてどのページから入るべきかを整理します。

## 先に準備すること

- すべての利用者は、まず [インストール](installation.md) を行います。
- 実機を使う Experiment と 低レベル API では、続けて [システム設定](system-configuration.md) を用意します。
- QuantumSimulator だけを使う場合は、実機向けの設定ファイルは不要です。

## 実機実験を進めたいなら Experiment

実機実験を高レベルなワークフローで進めたい場合の、推奨される入口です。
装置接続、測定実行、量子デバイスの特性評価、較正、ベンチマークまでを一貫して扱えます。

推奨する進み方:

1. [`Experiment` クラス](../experiment/index.md) で全体像を確認する
2. [クイックスタート](quickstart.md) で基本ワークフローを一通り試す
3. [Experiment サンプルワークフロー](../experiment/examples.md) で notebook をたどる
4. 必要に応じて [コミュニティ提供ワークフロー](contrib-workflows.md) を使う

## 実機なしで試したいなら QuantumSimulator

実機に接続せず、パルスレベルのダイナミクスやパルス設計をオフラインで試したい場合の入口です。
モデルの振る舞いを確かめたいときや、実機投入前に試行錯誤したいときに向いています。

推奨する進み方:

1. [QuantumSimulator](../simulator/index.md) で扱える範囲を確認する
2. 必要に応じて [パルスシーケンスの組み方](../pulse-sequences/index.md) を読む
3. [QuantumSimulator サンプルワークフロー](../simulator/examples.md) から notebook を始める

## measurement 側を直接扱いたいなら 低レベル API

session、schedule、capture/readout、sweep、backend integration などを
直接扱いたい場合の入口です。通常の実機実験では Experiment を使い、
measurement 側の概念を主役にしたいときだけこちらを選びます。

推奨する進み方:

1. [低レベル API 概要](../low-level-apis/index.md) でこのセクションの役割を確認する
2. 必要に応じて [パルスシーケンスの組み方](../pulse-sequences/index.md) を読む
3. [Measurement API 概要](../measurement/index.md) で API の中心概念を確認する
4. [Measurement サンプルワークフロー](../measurement/examples.md) から notebook を始める

## 補助的に読むページ

- [パルスシーケンスの組み方](../pulse-sequences/index.md): Experiment、QuantumSimulator、Measurement で共通して使う `PulseSchedule` の考え方を整理したいときに読みます。
- [サンプル集](../../examples/index.md): 目的別に notebook を探したいときに使います。

## 利用者ではなく開発者として参加する

Qubex のコードベース自体を拡張したい場合は、まず [Contributing](../../CONTRIBUTING.md) を読み、その後で [開発ガイド](../../developer-guide/index.md) に進んでください。
