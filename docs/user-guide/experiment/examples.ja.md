# Experiment サンプルワークフロー

このページでは、`Experiment` ワークフローの主要な notebook 入口を紹介します。
まず [クイックスタート](../getting-started/quickstart.md) を終えてから使ってください。

## 推奨する出発点

- [基本的な使い方](../../examples/experiment/0_basic_usage.ipynb): Experiment session を作成し、実機ベースの基本ワークフローを実行します。
- [分光](../../examples/experiment/1_spectroscopy.ipynb): 標準的な scan から共振器と量子ビットの周波数を求めます。
- [周波数較正](../../examples/experiment/2_frequency_calibration.ipynb): 駆動と読み出しの周波数を調整します。
- [パルス較正](../../examples/experiment/3_pulse_calibration.ipynb): gate operation に使う control pulse を調整します。
- [T1 / T2 実験](../../examples/experiment/4_t1_t2_experiments.ipynb): 緩和と位相緩和の標準実験で coherence を測定します。

## さらに進んだ Experiment ワークフロー

- [乱択ベンチマーキング](../../examples/experiment/5_randomized_benchmarking.ipynb): Clifford ベースのランダム系列で gate 性能を見積もります。
- [状態トモグラフィ](../../examples/experiment/6_state_tomography.ipynb): 繰り返し測定から量子状態を再構成します。
- [ef 特性評価](../../examples/experiment/7_ef_characterization.ipynb): 計算基底を超えた transmon の高次準位を扱います。
- [状態分類](../../examples/experiment/8_state_classification.ipynb): 読み出し状態分類器を学習・確認します。
- [CR 較正](../../examples/experiment/9_cr_calibration.ipynb): 2 量子ビット制御のための cross-resonance 相互作用を較正します。
- [DSP classification を使った mid-circuit measurement](../../examples/experiment/10_mid_circuit_dsp_classification.ipynb): 2 回の readout を含む single-shot 実験を実行し、結果が `gg` と `ee` に集まることを確認します。

## 関連ページ

- [パルスシーケンスの組み方](../pulse-sequences/index.md)
- [`Experiment` クラス](index.md)
- [コミュニティ提供ワークフロー](../getting-started/contrib-workflows.md)
- [サンプル集全体](../../examples/index.md)
