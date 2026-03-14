# 低レベル API

このセクションは、backend integration、readout utility、measurement 側の
tooling、sequencer 寄りの実行経路を扱う開発者向けです。

Qubex の多くの利用者は、実機では [Experiment](../experiment/index.md)、
オフラインでは [Simulator](../simulator/index.md) から始めるのが適切です。
session、schedule、capture/readout、sweep、backend 実行の詳細を主語に
したいときに、このセクションを使ってください。

## ここから始める

- [PulseSchedule の組み方](../pulse-sequences/index.md)
- [Measurement API 概要](../measurement/index.md)
- [Measurement サンプルワークフロー](../measurement/examples.md)

## 典型的な利用場面

- backend integration の実装や検証を行う
- readout 固有の utility や解析 helper を作る
- custom schedule、sweep、sequencer フローを開発する
- measurement 側のデータモデルや execution contract を直接扱う

## 次のような場合は Experiment を選ぶ

- 実機実験の多くで推奨されるユーザー向けワークフローを使いたい
- 評価、較正、ベンチマークの組み込みルーチンを使いたい
- セットアップ、実行、解析までを 1 つの facade で扱いたい

## 関連ドキュメント

- [始め方を選ぶ](../getting-started/choose-where-to-start.md)
- [サンプル集](../../examples/index.md)
- [開発ガイド](../../developer-guide/index.md)
