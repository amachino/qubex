# Simulator サンプルワークフロー

このページでは、`Simulator` ワークフローの主要な notebook 入口を紹介します。
実機ベースの実行に進む前に、パルスレベルのダイナミクスをオフラインで確認したいときに使ってください。

## 推奨する出発点

- [ラビ振動](../../examples/simulator/1_rabi_oscillation.ipynb): 単一の駆動された transmon から始めて、そのダイナミクスを確認します。
- [結合した 2 量子ビット](../../examples/simulator/2_coupled_qubits.ipynb): 結合した多体系での相互作用をシミュレーションします。
- [量子ビットと共振器](../../examples/simulator/3_qubit_resonator.ipynb): qubit-resonator 系の振る舞いをハミルトニアンレベルで調べます。
- [複数制御](../../examples/simulator/4_multi_control.ipynb): 1 つの simulation setup で複数 control channel を扱います。

## 較正寄りの解析

- [π パルス](../../examples/simulator/5_pi_pulse.ipynb): 目的の回転を実現する pulse を調整します。
- [DRAG 較正](../../examples/simulator/6_drag_calibration.ipynb): leakage 低減のための pulse shaping を試します。
- [CR ダイナミクス](../../examples/simulator/7_cr_dynamics.ipynb): cross-resonance のダイナミクスを simulation で確認します。
- [CR 較正](../../examples/simulator/8_cr_calibration.ipynb): 実機を使わずに 2 量子ビット較正ワークフローを反復します。
- [Jazz](../../examples/simulator/9_jazz.ipynb): さらに進んだ simulator 例を確認します。

## 関連ページ

- [Simulator 概要](index.md)
- [サンプル集全体](../../examples/index.md)
