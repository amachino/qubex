# Quickstart

This quickstart introduces the v1.5.0 `Experiment` workflow.
It assumes you already have configuration and parameter files for your chip.

> [!NOTE]
> Qubex loads configuration and parameter files that describe your chip, wiring, and control settings.
>
> - Default location: `/home/shared/qubex-config/<chip_id>/`.
> - Custom locations can be passed via `config_dir` and `params_dir` when creating an `Experiment` object.
> See [Configuration](../reference/configuration.md) for details.

## 1. Create an experiment

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

q0, q1 = exp.qubit_labels[:2]
```

## 2. Connect to hardware

```python
exp.connect()
```

## 3. Optionally update device settings

If you want to push the current configuration and parameter settings to the
devices, run:

```python
exp.configure()
```

## 4. Run a basic measurement with `measure`

```python
waveform = np.full(16, 0.01 + 0.01j)

result = exp.measure(
    sequence={
        q0: waveform,
        q1: waveform,
    },
    mode="avg",
    n_shots=1024,
)

result.plot()
print(result.data)
```

`measure()` lets you pass each qubit control waveform as a simple `numpy`
array. Qubex automatically appends the readout sequence after the control
waveforms, so you can start from a minimal waveform map.

## 5. Build a pulse sequence with `PulseSchedule`

```python
with qx.PulseSchedule([q0, q1]) as schedule:
    schedule.add(
        q0,
        qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
    )
    schedule.add(
        q1,
        qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
    )
    schedule.barrier()
    schedule.add(q0, qx.pulse.Blank(duration=32))
    schedule.add(q1, qx.pulse.Blank(duration=32))

schedule.plot()

sequence_result = exp.execute(
    schedule=schedule,
    mode="avg",
    n_shots=1024,
    final_measurement=True,
)

sequence_result.plot()
```

Use `add()` to place pulses on channels and `barrier()` to align channels before the next block.
This example uses `final_measurement=True` to append one readout automatically,
but `execute()` also lets you place readout pulses inside the schedule for
mid-circuit measurement. Because one target can be captured multiple times,
`sequence_result.data[target]` contains a list of capture results for that
target.

## 6. Sweep a parameter with `sweep_parameter`

```python
sweep = exp.sweep_parameter(
    sequence=lambda amplitude: {
        q0: qx.pulse.Rect(duration=64, amplitude=amplitude),
    },
    sweep_range=np.linspace(0.0, 0.08, 21),
    n_shots=1024,
    xlabel="Drive amplitude",
    ylabel="Readout response",
)

sweep.plot(normalize=True)
```

You can also return a `PulseSchedule` from a factory function and sweep that
schedule directly.

```python
def build_schedule(amplitude: float) -> qx.PulseSchedule:
    with qx.PulseSchedule([q0]) as schedule:
        schedule.add(
            q0,
            qx.pulse.Gaussian(duration=64, amplitude=amplitude, sigma=16),
        )
        schedule.add(q0, qx.pulse.Blank(duration=32))
    return schedule


schedule_sweep = exp.sweep_parameter(
    sequence=build_schedule,
    sweep_range=np.linspace(0.0, 0.08, 21),
    n_shots=1024,
    xlabel="Drive amplitude",
    ylabel="Readout response",
)

schedule_sweep.plot(normalize=True)
```

## Next steps

- Learn how configuration and targets are modeled: [Concepts](../concepts/overview.md)
- Continue with high-level experiment workflows: [Tutorials](../tutorials/index.md)
- Use lower-level APIs only when needed: [Basic measurement (Measurement API, advanced)](../how-to/basic-measurement.md)
- Learn more about reusable schedules: [Build a measurement schedule (Measurement API, advanced)](../how-to/run-pulse-schedule.md)
- Explore notebooks: [Examples](../../examples/index.md)
