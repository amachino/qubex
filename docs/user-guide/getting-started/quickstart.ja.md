# クイックスタート

このクイックスタートでは `Experiment` ワークフローを紹介します。
system に必要な設定ファイルと parameter ファイルがすでに用意されていることを前提にします。

> [!NOTE]
> Qubex は、chip、wiring、制御設定を記述した configuration file と parameter file を読み込みます。
>
> - 1 つの具体的な装置構成を選ぶ識別子には `system_id` を使ってください
> - `Experiment` を作成するときは `config_dir` と `params_dir` の両方を明示的に渡してください
> - これらのファイルは [システム設定](system-configuration.md) に従って作成してください
> - `Experiment` の基底単位は、時間系が `ns`、周波数系が `GHz` です

## 1. `Experiment` を作成する

system、対象 qubit、利用する configuration ディレクトリと parameter ディレクトリを指定して `Experiment` を作成します。`exp` があれば、装置接続、測定、パルススケジュールの実行、parameter sweep をメソッド経由で行えます。

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    system_id="SYSTEM_A",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)

Q0, Q1 = exp.qubit_labels[:2]
RQ0, RQ1 = exp.resonator_labels[:2]
```

## 2. 装置に接続する

測定やスケジュール実行の前に、設定された装置へ接続するため `connect()` を使います。これにより通信を確立し、リンク状態を確認し、装置側の現在設定をセッションへ取り込みます。

```python
exp.connect()
```

## 3. 必要なら装置設定を更新する

現在の設定ファイルや parameter を装置側へ反映したい場合にだけ `configure()` を使ってください。

```python
exp.configure()
```

> [!CAUTION]
> この操作は装置の状態を変更します。共有システムでは、同じ装置を利用している他ユーザーに影響する可能性があります。

## 4. `measure` で基本測定を実行する

制御波形を直接与え、読み出し波形およびキャプチャ設定を Qubex に自動付与させたい場合は `measure()` を使います。

```python
waveform = np.array([
    0.01 + 0.01j,
    0.01 + 0.01j,
    0.01 + 0.01j,
    0.01 + 0.01j,
])

sequence = {
    Q0: waveform,
    Q1: waveform,
}

result = exp.measure(
    sequence=sequence,
    mode="avg",
    n_shots=1024,
)
result.plot()
print("avg:", result.data[Q0].kerneled)

result = exp.measure(
    sequence=sequence,
    mode="single",
    n_shots=1024,
)
result.plot()
print("single:", result.data[Q0].kerneled)
```

`sequence` に含まれる各量子ビットに対して、Qubex は制御波形を適用し、対応する読み出し共振器へ読み出しパルスを送り、その反射信号を返します。`kerneled` は時間積分した反射信号を複素 I/Q データで表したものです。`avg` モードでは単一の複素数、`single` モードでは shot ごとの複素数配列になります。

## 5. `PulseSchedule` でパルスシーケンスを構築する

再利用可能な `Pulse` オブジェクトからパルスシーケンスを明示的に組み立てたい場合は `PulseSchedule` を作成します。

ワークフローに依らないスケジュール構築の説明は [パルスシーケンスの組み方](../pulse-sequences/index.md) を参照してください。

```python
pulse = qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16)
pulse.plot()

schedule = qx.PulseSchedule()
with schedule as s:
    s.add(Q0, pulse)
    s.add(Q0, pulse.scaled(2))
    s.barrier()
    s.add(Q1, pulse.shifted(np.pi / 6))

schedule.plot()
```

`PulseSchedule` を作成したら、`with` ブロック内で各 channel に対して `add()` を呼び、Pulse を追加します。次の block の前で channel を揃えたいときは `barrier()` を使います。`with` ブロックを抜けると、Qubex は全 channel を自動的に同じ長さまで pad します。この例ではパルスシーケンスだけを構築しており、`scaled()` や `shifted()` で同じベース `Pulse` から派生 `Pulse` を作る方法も示しています。

## 6. `sweep_parameter` で parameter を掃引する

1 つの系列を繰り返し実行しながら、ある parameter を範囲に沿って変えたい場合は `sweep_parameter()` を使います。`sweep_range` の各点ごとに、Qubex は系列を評価し、測定応答を `result.data[target]` に保存します。

```python
result = exp.sweep_parameter(
    sequence=lambda amplitude: {
        Q0: qx.pulse.Rect(duration=64, amplitude=amplitude),
    },
    sweep_range=np.linspace(0.0, 0.1, 21),
    n_shots=1024,
    xlabel="Drive amplitude",
    ylabel="Readout response",
)

result.plot()
print("sweep_range:", result.data[Q0].sweep_range)
print("data:", result.data[Q0].data)
```

factory function から `PulseSchedule` を返して、その schedule 自体を直接 sweep することもできます。これは T1 系列の待ち時間 sweep のように、blank duration を pulse sampling period に合わせたいときに便利です。以下では、対数間隔の待ち時間を有効な時間グリッドに離散化してから sweep しています。
以下の例では具体値として `2 ns` を使っていますが、実際には使用する装置に
合わせた sampling period を指定してください。

```python
wait_range = exp.util.discretize_time_range(
    np.geomspace(100, 100e3, 51),
    sampling_period=2,
)


def t1_sequence(wait: float) -> qx.PulseSchedule:
    schedule = qx.PulseSchedule()
    with schedule as s:
        s.add(Q0, qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16))
        s.add(Q0, qx.pulse.Blank(duration=wait))
    return schedule


result = exp.sweep_parameter(
    sequence=t1_sequence,
    sweep_range=wait_range,
    n_shots=1024,
    xlabel="Wait duration (ns)",
    ylabel="Readout response",
    xaxis_type="log",
)

result.plot()
print("sweep_range:", result.data[Q0].sweep_range)
print("data:", result.data[Q0].data)
```

## 7. `execute` で schedule を実行する

`PulseSchedule` をそのまま実行し、カスタム読み出しパルスを共振器チャネルに直接配置したい場合は `execute()` を使います。1 つの schedule に複数の読み出しイベントを含めたいときに便利です。

```python
control_pulse = qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16)
readout_pulse = qx.pulse.FlatTop(duration=256, amplitude=0.1, tau=32)

schedule = qx.PulseSchedule()
with schedule as s:
    s.add(RQ0, readout_pulse)
    s.barrier()
    s.add(Q0, qx.pulse.Blank(duration=128))
    s.barrier()
    s.add(Q0, control_pulse)
    s.barrier()
    s.add(RQ0, readout_pulse.scaled(0.8))

schedule.plot()

result = exp.execute(
    schedule=schedule,
    mode="avg",
    n_shots=1024,
)

result.plot()
print("n_captures:", len(result.data[Q0]))
```

この例では、`control_pulse` と `readout_pulse` を schedule 内で再利用しています。最初に 1 回読み出しを行い、その後 blank interval と control pulse を入れ、最後にもう 1 回読み出しを行います。`RQ0` は 2 回読み出しされるため、`result.data[Q0]` には 2 つのキャプチャ結果が入ります。

## 8. Experimental な `run_*` 非同期メソッド

`Experiment` には、`run_measurement()`、`run_sweep_measurement()`、
`run_ndsweep_measurement()` のような async-first な測定 API もあります。
これらは Experimental な機能として扱ってください。公開 API ではありますが、
将来のリリースでシグネチャや挙動が変わる可能性があります。

アプリケーション全体がすでに async で動いている場合に使ってください。
script では `asyncio.run(...)`、notebook では `await` を使います。

```python
import asyncio


async def main() -> None:
    result = await exp.run_measurement(
        schedule=schedule,
        n_shots=1024,
    )
    print(type(result).__name__)


asyncio.run(main())
```

現時点で安定性を優先するなら、ここまでで紹介した従来の同期メソッド
（`measure()`、`execute()`、`sweep_parameter()`）を使ってください。

## 次のステップ

- より広い `Experiment` ワークフローを見る: [`Experiment` クラス](../experiment/index.md)
- 整理された notebook 集へ進む: [Experiment サンプルワークフロー](../experiment/examples.md)
- より高度な contrib ルーチンを使う: [コミュニティ提供ワークフロー](contrib-workflows.md)
