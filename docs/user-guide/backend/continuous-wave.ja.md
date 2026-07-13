# QuEL-1 連続波出力

`Quel1BackendController` は、QuEL-1 の 1 つの AWG channel で長時間動作する
連続波を開始・停止できます。これは hardware 検証や CW 実験のための
controller-level capability です。`Measurement` や `Experiment` にはまだ公開していません。

## 使う場面

有限長の `MeasurementSchedule` 実行ではなく、mixer、周波数計画、analyzer trace、
意図的な aliasing などを確認したい場合に使います。

実行中の wavegen task は controller が保持します。停止は同じ controller instance
から行ってください。

実機 workflow として実行する場合は、guard 付きの
[QuEL-1 連続波 notebook](../../examples/backend/quel1_continuous_wave.ipynb) も参照してください。

## 出力を開始する

```python
from qubex.backend.quel1 import Quel1BackendController

controller = Quel1BackendController()
controller.define_clockmaster(ipaddr="10.3.0.x")
controller.define_box(
    box_name="S173R",
    ipaddr_wss="10.1.0.x",
    boxtype="quel1se-riken8",
)
controller.connect(box_names="S173R")

config = controller.start_continuous_wave(
    box_name="S173R",
    port=2,
    channel=0,
    awg_freq_hz=7_812_500.0 * 20,
    amplitude=0.2,
)
```

`awg_freq_hz` は AWG baseband 周波数です。QuEL-1 の CW 出力は 128 ns の
chunk を繰り返して生成するため、周波数は `1 / 128 ns = 7.8125 MHz` の整数倍で、
AWG baseband の Nyquist 範囲内である必要があります。

戻り値の `Quel1ContinuousWaveConfig` には、実際の AWG 周波数、chunk 内周期数、
波形名、ログ用に解決した LO/CNCO/FNCO、計算できた場合の出力周波数、repeat count、
名目上の duration が入ります。

## 現在の LO/NCO 設定を保つ

デフォルトでは、`start_continuous_wave()` は `config_port()` を呼ばず、FNCO も
更新しません。現在の hardware LO/CNCO/FNCO 設定を使い、`dump_port()` から読める値を
ログに出します。

`configure_port=False` のままでも `awg_freq_hz` は反映されます。この場合に変わるのは
生成する AWG waveform だけです。

このデフォルトは、CW 出力開始の副作用として port の位相基準を変えないためのものです。

## 出力設定を明示的に更新する

CW 開始時に出力設定も更新したい場合だけ、`configure_port=True` を渡します。
対象には LO/CNCO/FNCO、sideband、VATT、full-scale current、RF switch state を
含められます。

```python
config = controller.start_continuous_wave(
    box_name="S173R",
    port=2,
    channel=0,
    awg_freq_hz=0.0,
    amplitude=0.1,
    configure_port=True,
    lo_freq_hz=10_500_000_000,
    cnco_freq_hz=2_250_000_000,
    fnco_freq_hz=750_000_000,
    sideband="U",
    vatt=2048,
    fullscale_current=39000,
    rfswitch="pass",
)
```

`configure_port=True` なしで出力設定引数を指定した場合、Qubex は hardware に
触る前に `ValueError` を送出します。

## 出力を停止する

1 つの出力は同じ物理 key で停止します。

```python
stopped = controller.stop_continuous_wave(
    box_name="S173R",
    port=2,
    channel=0,
)
```

`stop_continuous_wave()` は、controller が記憶している task を停止・削除した場合に
`True`、該当 task がない場合に `False` を返します。

記憶しているすべての CW task を停止するには、次を使います。

```python
controller.stop_all_continuous_waves()
```

`disconnect()` も backend resource を解放する前に best-effort で
`stop_all_continuous_waves()` を呼びます。

異なる box 集合を指定して `connect()` し直す場合、QuEL-1 runtime state は作り直されます。
その場合、controller は reconnect の前に記憶している CW 出力を停止します。

## ログと alias warning

開始時には、LO/CNCO/FNCO/AWG の各周波数、`FNCO + AWG`、および
LO/CNCO/FNCO/sideband が揃っている場合の RF 出力周波数を GHz 単位で `INFO` level に
出します。

`abs(FNCO + AWG)` が 800 MHz を超える場合は、出力に aliasing が含まれる可能性が
あるため warning を出します。この条件は error ではありません。意図的な aliasing
テストは warning 後も継続できます。
