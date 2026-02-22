# Run a pulse schedule

This guide shows how to build a `qxpulse.PulseSchedule` and execute it with `Measurement`.

## Build a schedule

```python
from qxpulse import Blank, FlatTop, Gaussian, PulseSchedule
from qubex.measurement import Measurement

cli = Measurement(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

with PulseSchedule() as schedule:
    schedule.add("Q00", Gaussian(duration=32, amplitude=0.03, sigma=8))
    schedule.add("Q01", Gaussian(duration=32, amplitude=0.03, sigma=8))
    schedule.add("Q00", Blank(duration=16))
    schedule.barrier()
    schedule.add("RQ00", FlatTop(duration=256, amplitude=0.1, tau=32))
    schedule.add("RQ01", FlatTop(duration=256, amplitude=0.1, tau=32))
```

## Execute the schedule

```python
result = cli.execute(
    schedule,
    mode="single",
    shots=1024,
)

result.plot()
```

## Advanced: build and execute measurement schedules

```python
measurement_schedule = cli.build_measurement_schedule(pulse_schedule=schedule)
config = cli.create_measurement_config(mode="single", shots=1024)

result = cli.execute_measurement_schedule(
    schedule=measurement_schedule,
    config=config,
)
```
