# サンプル集

このセクションでは、ドキュメントに同梱されているレンダリング済み Jupyter Notebook へのリンクをまとめています。

Qubex が初めてなら、まず [始め方を選ぶ](../user-guide/getting-started/choose-where-to-start.md) から始めて、対応するガイドに進んでください。

- [`Experiment` クラス](../user-guide/experiment/index.md)
- [`QuantumSimulator` クラス](../user-guide/simulator/index.md)
- [低レベル API](../user-guide/low-level-apis/index.md)

## Experiment

- [基本的な使い方](experiment/0_basic_usage.ipynb)
- [分光](experiment/1_spectroscopy.ipynb)
- [周波数較正](experiment/2_frequency_calibration.ipynb)
- [パルス較正](experiment/3_pulse_calibration.ipynb)
- [T1 / T2 実験](experiment/4_t1_t2_experiments.ipynb)
- [ランダム化ベンチマーク](experiment/5_randomized_benchmarking.ipynb)
- [状態トモグラフィ](experiment/6_state_tomography.ipynb)
- [EF 特性評価](experiment/7_ef_characterization.ipynb)
- [状態分類](experiment/8_state_classification.ipynb)
- [CR 較正](experiment/9_cr_calibration.ipynb)

## 低レベル API

### `measurement`

- [Measurement config](measurement/measurement_config.ipynb)
- [Measurement session](measurement/measurement_client.ipynb)
- [Loopback capture](measurement/capture_loopback.ipynb)
- [Measurement sweep builder](measurement/sweep_measurement_builder.ipynb)
- [Sweep measurement executor](measurement/sweep_measurement_executor.ipynb)

### `system`

- [ConfigLoader Example](system/config_loader.ipynb)
- [QuEL-3 Experiment Configure Check](system/quel3_experiment_configure_check.ipynb)

### `backend`

- [PulseSchedule to QuEL-3 Sequencer Flow](measurement/quel3_sequencer_builder_flow.ipynb)
- [QuEL-3 Deploy Check](system/quel3_deploy_check.ipynb)

## QuantumSimulator

- [ラビ振動](simulator/1_rabi_oscillation.ipynb)
- [結合した 2 量子ビット](simulator/2_coupled_qubits.ipynb)
- [量子ビットと共振器](simulator/3_qubit_resonator.ipynb)
- [複数制御](simulator/4_multi_control.ipynb)
- [π パルス](simulator/5_pi_pulse.ipynb)
- [DRAG 較正](simulator/6_drag_calibration.ipynb)
- [CR ダイナミクス](simulator/7_cr_dynamics.ipynb)
- [CR 較正](simulator/8_cr_calibration.ipynb)
- [Jazz](simulator/9_jazz.ipynb)

## Pulse

- [パルスシーケンスの組み方](../user-guide/pulse-sequences/index.md)
- [Pulse チュートリアル](pulse/tutorial.ipynb)
- [Shape hash と waveform 再利用](pulse/shape_hash_and_waveform_reuse.ipynb)

## Analysis

- [フィッティング](analysis/fitting.ipynb)
- [プロット](analysis/plot.ipynb)
- [3D 回転](analysis/rotation3d.ipynb)

## Core

- [Async bridge](core/async_bridge.ipynb)
- [データモデル](core/data_model.ipynb)
- [Expression](core/expression.ipynb)
- [Model](core/model.ipynb)

## Clifford

- [Clifford の使い方](clifford/usage.ipynb)
- [Clifford マッピング](clifford/mapping.ipynb)
