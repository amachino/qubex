# Build pulse sequences with `PulseSchedule`

`PulseSchedule` is Qubex's shared container for pulse-level sequences.
It is a cross-cutting concept used by both `Experiment` and `QuantumSimulator`, and it
also appears in measurement-side workflows when pulse schedules are converted
into capture or execution requests.

Use this page when you want to learn how to build pulse sequences before
deciding whether they will be executed on hardware or used in offline studies.

## What `PulseSchedule` is for

- Describing time-ordered pulse events across one or more channels
- Reusing the same pulse objects and schedule patterns across hardware and simulation workflows
- Building a control sequence first, then deciding how to execute or analyze it

## Minimal pattern

Create a schedule, add pulses inside the `with` block, and use `barrier()` when
channels should align before the next block.

```python
import numpy as np
import qubex as qx

Q0 = "Q00"
Q1 = "Q01"

pulse = qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16)

schedule = qx.PulseSchedule()
with schedule as s:
    s.add(Q0, pulse)
    s.add(Q0, pulse.scaled(2.0))
    s.barrier()
    s.add(Q1, pulse.shifted(np.pi / 6))

schedule.plot()
```

## Core ideas

- `add(channel, pulse)`: place one pulse event on a channel
- `barrier()`: align channels before the next block of events
- Automatic padding: when the `with` block exits, channels are padded to the same duration
- Pulse reuse: derive related pulses from one base object with helpers such as `scaled()` and `shifted()`

## Where to use it next

- `Experiment`: pass a `PulseSchedule` to hardware-backed workflows such as the [Quickstart](../getting-started/quickstart.md)
- `QuantumSimulator`: reuse the same pulse objects and schedule-building style in [QuantumSimulator example workflows](../simulator/examples.md)
- `Low-level APIs`: convert pulse schedules into measurement-side flows from [Measurement API overview](../measurement/index.md)

## Learn more

- [Pulse tutorial notebook](../../examples/pulse/tutorial.ipynb)
- [Shape hash and waveform reuse](../../examples/pulse/shape_hash_and_waveform_reuse.ipynb)
- [Examples index](../../examples/index.md)
