# Quickstart

This quickstart introduces the `Experiment` workflow.
It assumes you already have configuration and parameter files for your chip.

> [!NOTE]
> Qubex loads configuration and parameter files that describe your chip, wiring, and control settings.
>
> - Default location: `/home/shared/qubex-config/<chip_id>/`
> - Custom locations: pass `config_dir` and `params_dir` when creating an `Experiment`
> - Base units in `Experiment`: time-like values use `ns`, and frequency-like values use `GHz`
>
> See [Configuration](../reference/configuration.md) for details.

## 1. Create an experiment

Create an `Experiment` by specifying the chip, target qubits, and the
configuration and parameter directories to use. Once you have `exp`, you can
connect to the instruments, run measurements, execute pulse schedules, and
sweep parameters through its methods.

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=[0, 1],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

Q0, Q1 = exp.qubit_labels[:2]
RQ0, RQ1 = exp.resonator_labels[:2]
```

## 2. Connect to instruments

Use `connect()` to connect to the configured instruments before running
measurements or schedules. It establishes communication, checks link status,
and pulls the current instrument-side settings into the session.

```python
exp.connect()
```

## 3. Optionally update instrument settings

Use `configure()` only when you want to push the current configuration and
parameter settings to the instruments.

```python
exp.configure()
```

> [!CAUTION]
> This operation changes the state of the instruments. On shared systems, it
> can affect other users who are using the same instruments.

## 4. Run a basic measurement with `measure`

Use `measure()` when you want to provide control waveforms directly and let
Qubex append the readout automatically.

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

For each qubit in `sequence`, Qubex applies the control waveform, sends a
readout pulse to the corresponding readout resonator, and returns the
reflected signal. `kerneled` is the time-integrated reflected signal expressed
as complex I/Q data: in `avg` mode it is a single complex value, and in
`single` mode it is a complex array with one value per shot.

## 5. Build a pulse sequence with `PulseSchedule`

Create a `PulseSchedule` when you want to build a control sequence explicitly
from reusable pulse objects.

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

After creating a `PulseSchedule`, add pulses inside the `with` block by calling
`add()` on each channel. Use `barrier()` to align channels before the next
block, and when the block exits, Qubex pads all channels to the same length
automatically. This example only builds the control sequence, and it also
shows how to derive related pulses from the same base pulse with `scaled()`
and `shifted()`.

## 6. Sweep a parameter with `sweep_parameter`

Use `sweep_parameter()` when you want to rerun the same sequence while
changing one parameter across a range of values. For each point in
`sweep_range`, Qubex evaluates the sequence and stores the measured response in
`result.data[target]`.

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

You can also return a `PulseSchedule` from a factory function and sweep that
schedule directly. This is useful for wait-time sweeps in T1-like sequences,
where the blank duration should follow the pulse sampling period. Here,
the log-spaced wait values are discretized onto the valid time grid before the
sweep.

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

## 7. Execute a schedule with `execute`

Use `execute()` when you want to run a `PulseSchedule` as written and place
custom readout pulses directly on resonator channels. This is useful when one
schedule should contain multiple readout events.

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

This example reuses `control_pulse` and `readout_pulse` inside the schedule.
It first performs a readout, then inserts a blank interval and a control
pulse, and finally performs a second readout. Because `RQ0` is read out twice,
`result.data[Q0]` contains two capture results.

## Next steps

- Explore notebooks: [Examples](../../examples/index.md)
- Review configuration requirements: [Configuration](../reference/configuration.md)
- Use lower-level APIs only when needed: [Basic measurement](../how-to/basic-measurement.md)
