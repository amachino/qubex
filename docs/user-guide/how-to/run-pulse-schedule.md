# Run a pulse schedule (Measurement API, advanced)

This guide shows how to build a `qxpulse.PulseSchedule` and execute it with the lower-level `Measurement` API.

For most workflows, use `Experiment.execute()` and related high-level APIs.

## Build a schedule

```python
from qxpulse import Blank, FlatTop, Gaussian, PulseSchedule
from qubex.measurement import Measurement

session = Measurement(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

schedule = PulseSchedule()
with schedule as s:
    s.add("Q00", Gaussian(duration=32, amplitude=0.03, sigma=8))
    s.add("Q01", Gaussian(duration=32, amplitude=0.03, sigma=8))
    s.add("Q00", Blank(duration=16))
    s.barrier()
    s.add("RQ00", FlatTop(duration=256, amplitude=0.1, tau=32))
    s.add("RQ01", FlatTop(duration=256, amplitude=0.1, tau=32))
```

## Execute the schedule

```python
result = session.execute(
    schedule,
    mode="single",
    shots=1024,
)

result.plot()
```

## Advanced: build and execute measurement schedules

```python
measurement_schedule = session.build_measurement_schedule(pulse_schedule=schedule)
config = session.create_measurement_config(
    mode="single",
    shots=1024,
    enable_dsp_demodulation=True,
    enable_dsp_sum=True,
    enable_dsp_classification=False,
    line_param0=(1.0, 0.0, 0.0),
    line_param1=(0.0, 1.0, 0.0),
)

result = await session.run_measurement(
    schedule=measurement_schedule,
    config=config,
)
```
